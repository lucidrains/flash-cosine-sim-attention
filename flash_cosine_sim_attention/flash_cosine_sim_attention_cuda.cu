#include <cassert>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>
#include "dispatch.h"

// error handler
// from https://leimao.github.io/blog/Proper-CUDA-Error-Checking

#define CHECK_LAST_CUDA_ERROR(cuda_sync) check(__FILE__, __LINE__, cuda_sync)
void check(const char* file, const int line, const bool cuda_sync)
{
    if (cuda_sync)
        cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

#define ACCESSOR(x, n, type) x.packed_accessor32<type, n, torch::RestrictPtrTraits>()

// type alias

template <typename scalar_t, int dims>
using PackedAccessor = torch::PackedTensorAccessor32<scalar_t, dims, torch::RestrictPtrTraits>;

// helper functions

__host__ __device__ int cdiv(int numer, int denom) {
    return (numer + denom - 1) / denom;
}

// shared memory struct

namespace mem {
    template<typename T, int N_tile, int M_tile>
    struct shared_fragment {
        static constexpr int N = N_tile;
        static constexpr int M = M_tile;
        static constexpr int stride = M + (sizeof(T) == 2 ? 8 : 1);
        static constexpr int size = N * stride;

        T* smem;

        __device__ shared_fragment(char* shared_base)
          : smem(reinterpret_cast<T*>(shared_base)) { }


        __device__ T& operator()(int x, int y) {
            return smem[y * stride + x];
        }

        template<typename accessor>
        __device__ void load(accessor gmem, int tile_x, int tile_y, int max_x, int max_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                int gx = x + tile_x;
                int gy = y + tile_y;
                int s_ind = y * stride + x;

                if ((max_x > 0  && gx >= max_x) || (max_y > 0 && gy >= max_y)) {
                    smem[s_ind] = 0.f;
                    continue;
                }

                smem[s_ind] = gmem[gy][gx];
            }
        }

        template<typename accessor>
        __device__ void load_transpose(accessor gmem, int tile_x, int tile_y, int max_x, int max_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % N;
                int y = i / N;
                int gx = x + tile_x;
                int gy = y + tile_y;
                int s_ind = x * stride + y;

                if ((max_x > 0  && gx >= max_x) || (max_y > 0 && gy >= max_y)) {
                    smem[s_ind] = 0.f;
                    continue;
                }

                smem[s_ind] = gmem[gy][gx];
            }
        }

        template<typename accessor>
        __device__ void store(accessor gmem, int tile_x, int tile_y, int max_x, int max_y) {
            for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
                int x = i % M;
                int y = i / M;
                int gx = x + tile_x;
                int gy = y + tile_y;
                int s_ind = y * stride + x;

                if ((max_x > 0  && gx >= max_x) || (max_y > 0 && gy >= max_y))
                    continue;

                gmem[gy][gx] = smem[s_ind];
            }
        }

        template<typename accessor>
        __device__ void load(accessor gmem, int tile_x, int tile_y) {
            load(gmem, tile_x, tile_y, 0, 0);
        }

        template<typename accessor>
        __device__ void store(accessor gmem, int tile_x, int tile_y) {
            store(gmem, tile_x, tile_y, 0, 0);
        }

        template<typename accessor>
        __device__ void load_transpose(accessor gmem, int tile_x, int tile_y) {
            load_transpose(gmem, tile_x, tile_y, 0, 0);
        }

        __device__ char* next() {
            return reinterpret_cast<char*>(smem + size);
        }

        // allow for storing rows

        template<typename accessor>
        __device__ void store_row(accessor gmem, int row_offset, int row_tile_size, int row_max) {
            for (int i = threadIdx.x; i < row_tile_size; i += blockDim.x) {
                int global_row = row_offset + i;

                if (global_row >= row_max)
                    continue;

                smem[i] = gmem[global_row];
            }
        }
    };
}

// rowsum accumulator

template <typename scalar_t, typename warp_tile_t, typename out_warp_tile_t>
struct rowsum_accumulator {
    static constexpr int N_tile = warp_tile_t::N_tile;
    static constexpr int M_tile = warp_tile_t::M_tile;

    float acc;

    __device__ void zero() {
        acc = 0;
    }

    template<typename shared_fragment>
    __device__ void add(shared_fragment& smem, int col_tile_offset, int col_seq_len) {
        if (threadIdx.x >= N_tile)
            return;

        #pragma unroll
        for (int i = 0; i < M_tile; i++) {
            if ((col_tile_offset + i) >= col_seq_len)
                continue;

            acc += smem(threadIdx.x, i);
        }
    }    

    template<typename F>
    __device__ void pointwise(F&& op) {
        if (threadIdx.x >= N_tile)
            return;

        acc = op(acc);
    }

    __device__ void multiply(float* smem, out_warp_tile_t& mma) {
        if (threadIdx.x < N_tile) smem[threadIdx.x] = acc;

        __syncthreads();

        mma.pointwise([&](scalar_t el, int, int y) {
            return (scalar_t) (el * smem[y]);
        });

        __syncthreads();
    }

    template<typename accessor>
    __device__ void store(accessor gmem, int row_tile_offset, int row_seq_len) {
        int row = row_tile_offset + threadIdx.x;

        if (threadIdx.x >= N_tile || row >= row_seq_len)
            return;

        gmem[row] = acc;
    }
};

// layout for every type of head dimension, not generalized yet

namespace layout {

    static constexpr int chunk_size = 16;

    // threads per block

    template<typename scalar_t, int dim_head>
    struct tpb {
        static constexpr int TPB = 256;
    };

    template<int dim_head>
    struct tpb<c10::Half, dim_head> {
        static constexpr int TPB = 256;
    };

    template<>
    struct tpb<c10::Half, 32> {
        static constexpr int TPB = 128;
    };

    // shared memory sizes that depends on layout, if needed

    template<typename scalar_t, int tile_size, int dim_head>
    struct smem {

        static constexpr int forward_size = (
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // q
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // k
            mem::shared_fragment<scalar_t, tile_size, tile_size>::size      // c
        );

        static constexpr int backward_size = (
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +    // q
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +    // k
            mem::shared_fragment<scalar_t, tile_size, tile_size>::size +     // c
            mem::shared_fragment<scalar_t, 1, tile_size>::size               // d
        );
    };

    template<typename scalar_t, int tile_size>
    struct smem<scalar_t, tile_size, 96> {

        static constexpr int forward_size = (
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // q
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // k
            mem::shared_fragment<scalar_t, tile_size, 96>::size             // c
        );

        static constexpr int backward_size = (
            mem::shared_fragment<scalar_t, chunk_size, 96>::size +          // q
            mem::shared_fragment<scalar_t, chunk_size, 96>::size +          // k
            mem::shared_fragment<scalar_t, tile_size, tile_size>::size +    // c
            mem::shared_fragment<scalar_t, 1, tile_size>::size              // d
        );
    };

    template<typename scalar_t, int tile_size>
    struct smem<scalar_t, tile_size, 128> {

        static constexpr int forward_size = (
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // q
            mem::shared_fragment<scalar_t, chunk_size, tile_size>::size +   // k
            mem::shared_fragment<scalar_t, tile_size, 128>::size            // c
        );

        static constexpr int backward_size = (
            mem::shared_fragment<scalar_t, chunk_size, 128>::size +          // q
            mem::shared_fragment<scalar_t, chunk_size, 128>::size +          // k
            mem::shared_fragment<scalar_t, tile_size, tile_size>::size +     // c
            mem::shared_fragment<scalar_t, 1, tile_size>::size               // d
        );
    };

    // f32

    template<typename scalar_t, int dim_head, int N_tile_, int M_tile_>
    struct warp {
        static constexpr int K_tile = 1;

        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        static constexpr int N_warp = 8;
        static constexpr int M_warp = 4;

        static constexpr int N_thread = N_tile_ / (N_warp * N_block);
        static constexpr int M_thread = M_tile_ / (M_warp * M_block);
    };


    template<typename scalar_t>
    struct warp<scalar_t, 64, 64, 64> {
        static constexpr int N_warp = 8;
        static constexpr int M_warp = 4;

        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        static constexpr int N_thread = 4;
        static constexpr int M_thread = 4;

        // constraints
        static_assert(N_warp * M_warp == 32);
        static_assert(N_block * M_block * N_warp * M_warp == layout::tpb<scalar_t, 64>::TPB);

        static_assert(N_warp * N_block * N_thread == 64);
        static_assert(M_warp * M_block * M_thread == 64);
    };

    // f16

    template<int dim_head, int N_tile_, int M_tile_>
    struct warp<c10::Half, dim_head, N_tile_, M_tile_> {
        static constexpr int K_tile = 16;

        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        static constexpr int N_thread = N_tile_ / (N_warp * N_block);
        static constexpr int M_thread = M_tile_ / (M_warp * M_block);
    };

    template<>
    struct warp<c10::Half, 64, 64, 64> {
        static constexpr int N_thread = 2;
        static constexpr int M_thread = 1;

        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        static constexpr int K_tile = 16;

        // constraints
        static_assert((N_warp == 16 && M_warp == 16) || (N_warp == 32 && M_warp == 8) || (N_warp == 8 && M_warp == 32));
        static_assert(N_block * M_block * 32 == layout::tpb<c10::Half, 64>::TPB);

        static_assert(N_thread * N_warp * N_block == 64);
        static_assert(M_thread * M_warp * M_block == 64);
    };

    template<>
    struct warp<c10::Half, 96, 64, 96> {
        static constexpr int N_thread = 1;
        static constexpr int M_thread = 3;

        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        static constexpr int N_block = 4;
        static constexpr int M_block = 2;

        static constexpr int K_tile = 16;

        // constraints
        static_assert((N_warp == 16 && M_warp == 16) || (N_warp == 32 && M_warp == 8) || (N_warp == 8 && M_warp == 32));
        static_assert(N_block * M_block * 32 == layout::tpb<c10::Half, 96>::TPB);

        static_assert(N_thread * N_warp * N_block == 64);
        static_assert(M_thread * M_warp * M_block == 96);
    };

    template<>
    struct warp<c10::Half, 32, 64, 32> {
        static constexpr int N_thread = 2;
        static constexpr int M_thread = 1;

        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        static constexpr int N_block = 2;
        static constexpr int M_block = 2;

        static constexpr int K_tile = 16;

        // constraints
        static_assert((N_warp == 16 && M_warp == 16) || (N_warp == 32 && M_warp == 8) || (N_warp == 8 && M_warp == 32));
        static_assert(N_block * M_block * 32 == layout::tpb<c10::Half, 32>::TPB);

        static_assert(N_thread * N_warp * N_block == 64);
        static_assert(M_thread * M_warp * M_block == 32);
    };

    template<>
    struct warp<c10::Half, 32, 64, 64> {
        static constexpr int N_thread = 2;
        static constexpr int M_thread = 2;

        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        static constexpr int N_block = 2;
        static constexpr int M_block = 2;

        static constexpr int K_tile = 16;

        // constraints
        static_assert((N_warp == 16 && M_warp == 16) || (N_warp == 32 && M_warp == 8) || (N_warp == 8 && M_warp == 32));
        static_assert(N_block * M_block * 32 == layout::tpb<c10::Half, 32>::TPB);

        static_assert(N_thread * N_warp * N_block == 64);
        static_assert(M_thread * M_warp * M_block == 64);
    };
}

// warp tile, done by @ahennequ Arthur Hennequin

#include <mma.h>

namespace mma {
    template<typename scalar_t, int dim_head, int N_tile_, int M_tile_>
    struct warp_tile {

        using l = layout::warp<scalar_t, dim_head, N_tile_, M_tile_>;

        // How much data is processed by a single thread:
        static constexpr int N_thread = l::N_thread;
        static constexpr int M_thread = l::M_thread;

        // Thread layout within a warp:
        static constexpr int N_warp = l::N_warp;
        static constexpr int M_warp = l::M_warp;

        // Warp layout within a block:
        static constexpr int N_block = l::N_block;
        static constexpr int M_block = l::M_block;

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_tile_;
        static constexpr int M_tile = M_tile_;
        static constexpr int K_tile = 1;

        // Registers:
        float C_frag[N_thread * M_thread]; // N x M fragment

        int warp_x;   // x offset of the warp within the block tile
        int warp_y;   // y offset of the warp within the block tile
        int thread_x; // x offset of the thread within the warp tile
        int thread_y; // y offset of the thread within the warp tile

        __device__ warp_tile() {
            int warp_id = threadIdx.x / 32;
            warp_x = (warp_id % M_block);
            warp_y = (warp_id / M_block);

            int lane_id = threadIdx.x & 31;
            thread_x = warp_x * M_warp * M_thread + lane_id % M_warp;
            thread_y = warp_y * N_warp * N_thread + lane_id / M_warp;
        }

        // Initialize C to all zeros
        __device__ void zero() {
            #pragma unroll
            for (int i = 0; i < N_thread * M_thread; i++) {
                C_frag[i] = 0.f;
            }
        }

        // Performs C = A * B + C
        template<typename fragA, typename fragB>
        __device__ void mma(fragA& A_sm, fragB& B_sm, int ka0, int kb0, int D) {
            float A_frag[N_thread]; // N x 1 fragment
            float B_frag[M_thread]; // 1 x M fragment

            for (int k = 0; k < D; k += K_tile) {
                // Load a N x 1 fragment of A from shared memory to registers:
                #pragma unroll
                for (int i = 0; i < N_thread; i++) {
                    A_frag[i] = A_sm(i * N_warp + thread_y, ka0 + k);
                }

                // Load a 1 x M fragment of B from shared memory to registers:
                #pragma unroll
                for (int i = 0; i < M_thread; i++) {
                    B_frag[i] = B_sm(i * M_warp + thread_x, kb0 + k);
                }

                // Compute:
                #pragma unroll
                for (int i = 0; i < N_thread; i++) {
                    #pragma unroll
                    for (int j = 0; j < M_thread ; j++) {
                        C_frag[i * M_thread + j] += A_frag[i] * B_frag[j];
                    }
                }
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int row = i * N_warp + thread_y;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int col = j * M_warp  + thread_x;
                    C_frag[i * M_thread + j] = op(C_frag[i * M_thread + j], col, row);
                }
            }
        }

        // Copy C from registers to shared memory
        template<typename shared_fragment>
        __device__ void store(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm(j * M_warp + thread_x, i * N_warp + thread_y) = C_frag[i * M_thread + j];
                }
            }
        }

        template<typename shared_fragment>
        __device__ void store_transpose(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    C_sm(i * N_warp + thread_y, j * M_warp + thread_x) = C_frag[i * M_thread + j];
                }
            }
        }

        // Stream C from registers to global memory using temporary shared memory buffer
        template<typename accessor, typename shared_fragment>
        __device__ void store(accessor gmem, shared_fragment& smem, int tile_x, int tile_y, int max_x, int max_y) {
            store(smem);
            __syncthreads();
            smem.store(gmem, tile_x, tile_y, max_x, max_y);
        }

        // atomic add C from registers to gmem
        template<typename accessor>
        __device__ void atomic_add(accessor gmem, int row_tile_offset, int col_tile_offset, int row_max, int col_max) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = i * N_warp + thread_y;
                int global_row = y + row_tile_offset;
                if (global_row >= row_max)
                    continue;

                #pragma unroll
                for (int j = 0; j < M_thread ; j++) {
                    int x = j * M_warp + thread_x;
                    int global_col = x + col_tile_offset;
                    if (global_col >= col_max)
                        continue;

                    atomicAdd((float*) &gmem[global_row][global_col], C_frag[i * M_thread + j]);
                }
            }
        }
    };

    using namespace nvcuda;
    template<int dim_head, int N_tile_, int M_tile_>
    struct warp_tile<c10::Half, dim_head, N_tile_, M_tile_> {

        using l = layout::warp<c10::Half, dim_head, N_tile_, M_tile_>;

        // How much data is processed by a single thread:
        static constexpr int N_thread = l::N_thread;
        static constexpr int M_thread = l::M_thread;

        // Thread layout within a warp:
        static constexpr int N_warp = l::N_warp;
        static constexpr int M_warp = l::M_warp;

        // Warp layout within a block:
        static constexpr int N_block = l::N_block;
        static constexpr int M_block = l::M_block;

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_tile_;
        static constexpr int M_tile = M_tile_;
        static constexpr int K_tile = 16;

        using output_t = half;

        // Registers:
        wmma::fragment<wmma::accumulator, N_warp, M_warp, K_tile, half> C_frag[N_thread * M_thread];

        int warp_x;   // x offset of the warp within the block tile
        int warp_y;   // y offset of the warp within the block tile

        __device__ warp_tile() {
            int warp_id = threadIdx.x / 32;
            warp_x = (warp_id % M_block);
            warp_y = (warp_id / M_block);
        }

        // Initialize C to all zeros
        __device__ void zero() {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    #pragma unroll
                    for (int k = 0; k < C_frag[i * M_thread + j].num_elements; k++) {
                        C_frag[i * M_thread + j].x[k] = (c10::Half) 0.f;
                    }
                }
            }
        }

        // Performs C = A * B + C
        template<typename fragA, typename fragB>
        __device__ void mma(fragA& A_sm, fragB& B_sm, int ka0, int kb0, int D) {
            wmma::fragment<wmma::matrix_a, N_warp, M_warp, K_tile, half, wmma::col_major> A_frag;
            wmma::fragment<wmma::matrix_b, N_warp, M_warp, K_tile, half, wmma::row_major> B_frag;

            for (int k = 0; k < D; k += K_tile) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    // Load a 1 x M fragment of B from shared memory to registers:
                    int x = (warp_x * M_thread + j) * M_warp;
                    wmma::load_matrix_sync(B_frag, reinterpret_cast<const half*>(&B_sm(x, kb0 + k)), B_sm.stride);

                    #pragma unroll
                    for (int i = 0; i < N_thread; i++) {
                        // Load a N x 1 fragment of A from shared memory to registers:
                        int y = (warp_y * N_thread + i) * N_warp;
                        wmma::load_matrix_sync(A_frag, reinterpret_cast<const half*>(&A_sm(y, ka0 + k)), A_sm.stride);

                        // Compute:
                        wmma::mma_sync(C_frag[i * M_thread + j], A_frag, B_frag, C_frag[i * M_thread + j]);
                    }
                }
            }
        }

        __device__ int get_warp_row(int i) {
            int tid = threadIdx.x % 32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
                if (std::is_same<output_t, half>::value) {
                    return (tid & 3) + ((tid & 4) << 1) + ((tid & 16) >> 2);
                } else {
                    return (tid & 16) / 4 + 2 * (tid & 4) + (tid & 1) + (i & 2);
                }
            #else
                return (i & 2) * 4 + tid / 4;
            #endif
        }

        __device__ int get_warp_col(int i) {
            int tid = threadIdx.x % 32;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
                if (std::is_same<output_t, half>::value) {
                    return (i & 7) + (tid & 8);
                } else {
                    return (tid & 10) + (i & 5);
                }
            #else
                return (tid % 4) * 2 + i % 2 + (i & 4) * 2;
            #endif
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    #pragma unroll
                    for (int k = 0; k < C_frag[i * M_thread + j].num_elements; k++) {
                        int col = get_warp_col(k) + (warp_x * M_thread + j) * M_warp;
                        int row = get_warp_row(k) + (warp_y * N_thread + i) * N_warp;
                        C_frag[i * M_thread + j].x[k] = op(C_frag[i * M_thread + j].x[k], col, row);
                    }
                }
            }
        }

        // Copy C from registers to shared memory
        template<typename shared_fragment>
        __device__ void store(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    #pragma unroll
                    for (int k = 0; k < C_frag[i * M_thread + j].num_elements; k++) {
                        int col = get_warp_col(k) + (warp_x * M_thread + j) * M_warp;
                        int row = get_warp_row(k) + (warp_y * N_thread + i) * N_warp;
                        C_sm(col, row) = C_frag[i * M_thread + j].x[k];
                    }
                }
            }
        }

        template<typename shared_fragment>
        __device__ void store_transpose(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    #pragma unroll
                    for (int k = 0; k < C_frag[i * M_thread + j].num_elements; k++) {
                        int col = get_warp_col(k) + (warp_x * M_thread + j) * M_warp;
                        int row = get_warp_row(k) + (warp_y * N_thread + i) * N_warp;
                        C_sm(row, col) = C_frag[i * M_thread + j].x[k];
                    }
                }
            }
        }

        // Stream C from registers to global memory using temporary shared memory buffer
        template<typename accessor, typename shared_fragment>
        __device__ void store(accessor gmem, shared_fragment& smem, int tile_x, int tile_y, int max_x, int max_y) {
            store(smem);
            __syncthreads();
            smem.store(gmem, tile_x, tile_y, max_x, max_y);
        }

        // atomic add C from registers to gmem
        template<typename accessor>
        __device__ void atomic_add(accessor gmem, int row_tile_offset, int col_tile_offset, int row_max, int col_max) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    #pragma unroll
                    for (int k = 0; k < C_frag[i * M_thread + j].num_elements; k++) {
                        int col = col_tile_offset + get_warp_col(k) + (warp_x * M_thread + j) * M_warp;
                        int row = row_tile_offset + get_warp_row(k) + (warp_y * N_thread + i) * N_warp;

                        if (col >= col_max || row >= row_max)
                            continue;

                        atomicAdd((float*) &gmem[row][col], __half2float(C_frag[i * M_thread + j].x[k]));
                    }
                }
            }
        }
    };
}

// forward kernel

template <typename scalar_t, int tile_size, int dim_head>
__global__ void forward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
          PackedAccessor<scalar_t, 4> o,
          PackedAccessor<float, 3> inv_l,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<scalar_t, 3> attn_bias,
    const float scale,
    const bool causal,
    const bool has_mask,
    const bool has_attn_bias,
    const bool attn_bias_batch_dim,
    const bool need_store_rowsum,
    const bool is_single_head_kv
) {
    // dimensions

    const int batch = blockIdx.y;
    const int heads = blockIdx.z;
    const int kv_heads = is_single_head_kv ? 0 : heads;

    const int row_seq_len = q.size(2);
    const int col_seq_len = k.size(2);
    const int seq_len_diff = col_seq_len - row_seq_len;

    constexpr int chunk_size = 16;
    const int row_tile_offset = blockIdx.x * tile_size;

    // registers

    using QK_mma_t  = mma::warp_tile<scalar_t, dim_head, tile_size, tile_size>;
    using out_mma_t = mma::warp_tile<scalar_t, dim_head, tile_size, dim_head>;

    float bias;

    QK_mma_t  QK_mma;
    out_mma_t out_mma;
    rowsum_accumulator<scalar_t, QK_mma_t, out_mma_t> L_acc;

    // shared memory

    using Q_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using K_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using V_sm_t = mem::shared_fragment<scalar_t, chunk_size, dim_head>;

    using C_sm_t = mem::shared_fragment<scalar_t, tile_size, tile_size>;
    using mask_sm_t = mem::shared_fragment<bool, 2, tile_size>;
    using L_sm_t = mem::shared_fragment<float, tile_size, 1>;
    using O_sm_t = mem::shared_fragment<scalar_t, tile_size, dim_head>;

    __shared__ scalar_t _shared_mem[layout::smem<scalar_t, tile_size, dim_head>::forward_size];

    auto __shared_mem = reinterpret_cast<char*>(_shared_mem);

    Q_sm_t Q_sm{__shared_mem};
    V_sm_t V_sm{__shared_mem};
    K_sm_t K_sm{Q_sm.next()};
    C_sm_t C_sm{K_sm.next()};
    L_sm_t L_sm{K_sm.next()};
    O_sm_t O_sm{K_sm.next()};
    mask_sm_t mask_sm{K_sm.next()};

    // shortcut accessors

    auto Q_ = q[batch][heads];
    auto K_ = k[batch][kv_heads];
    auto V_ = v[batch][kv_heads];
    auto l_ = inv_l[batch][heads];
    auto O_ = o[batch][heads];
    auto mask_ = mask[batch];
    auto bias_ = attn_bias[attn_bias_batch_dim ? batch : heads];

    // renamed vars

    auto col_tile_size = tile_size;
    auto row_tile_size = tile_size;

    // zero accumulators

    out_mma.zero();
    L_acc.zero();

    // loop over column tiles

    for (int col_tile_offset = 0; col_tile_offset < col_seq_len; col_tile_offset += col_tile_size) {
        if (causal && ((col_tile_offset - seq_len_diff) >= (row_tile_offset + row_tile_size)))
            continue;

        QK_mma.zero();

        // get qk similarity matrix

        for (int i = 0; i < dim_head; i += chunk_size) {
            Q_sm.load_transpose(Q_, i, row_tile_offset, 0, row_seq_len);
            K_sm.load_transpose(K_, i, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(Q_sm, K_sm, 0, 0, chunk_size);
            __syncthreads();
        }

        // store mask into smem if needed

        if (has_mask)
            mask_sm.store_row(mask_, col_tile_offset, col_tile_size, col_seq_len);

        __syncthreads();

        // scale and exponentiation, depending on masking or not

        QK_mma.pointwise([&](scalar_t el, int col, int row) -> scalar_t {
            int attn_col = col + col_tile_offset;
            int attn_row = row + row_tile_offset;

            if ((attn_col >= col_seq_len) ||
                (attn_row >= row_seq_len) ||
                (causal && ((attn_col - seq_len_diff) > attn_row)) ||
                (has_mask && !mask_sm.smem[col]))
                return 0.f;

            float shift = scale;

            if (causal && seq_len_diff == 0)
                if (attn_row == 0)
                    return 1.f;

                shift = min(shift, (float) attn_row);

            bias = has_attn_bias ? (float) bias_[attn_row][attn_col] : 0.f;

            return __expf(scale * el + bias - shift);
        });

        if (has_mask)
            __syncthreads();

        QK_mma.store_transpose(C_sm);

        __syncthreads();

        L_acc.add(C_sm, col_tile_offset, col_seq_len);

        // aggregate values with attention matrix

        for (int i = 0; i < col_tile_size; i += chunk_size) {
            V_sm.load(V_, 0, col_tile_offset + i, 0, col_seq_len);
            __syncthreads();

            out_mma.mma(C_sm, V_sm, i, 0, chunk_size);
            __syncthreads();
        }
    }

    L_acc.pointwise([](float el) { return 1.f / max(el, 1e-10); }); // get inverse of rowsums

    if (need_store_rowsum)
        L_acc.store(l_, row_tile_offset, row_seq_len);

    L_acc.multiply(L_sm.smem, out_mma);

    out_mma.store(O_, O_sm, 0, row_tile_offset, 0, row_seq_len);
}

// backwards preprocess

// delta = rowsum(do * o)
// do_scaled is moved into backwards kernel since seeing numerical issues unique to cosine sim attention

// done by @ptillet and @tridao for the triton fused attention and flash attention cuda impl

template <typename scalar_t, int dim_head>
__global__ void backward_preprocess(
    const PackedAccessor<scalar_t, 4> d_out,
    const PackedAccessor<scalar_t, 4> o,
          PackedAccessor<scalar_t, 3> delta
) {
    const int heads = o.size(1);

    const int batch_idx = blockIdx.x / heads;
    const int head_idx = blockIdx.x % heads;
    const int seq_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;

    const unsigned mask = __ballot_sync(0xFFFFFFFFU, dim_idx < dim_head);

    // registers

    float val = 0.0f;

    // shared memory

    __shared__ scalar_t _shared_mem_preprocess[dim_head / 32];

    scalar_t* sm_delta  = reinterpret_cast<scalar_t*>(&_shared_mem_preprocess);

    // global mem accessors

    auto do_ = d_out[batch_idx][head_idx][seq_idx];
    auto o_ = o[batch_idx][head_idx][seq_idx];
    auto delta_ = delta[batch_idx][head_idx];

    // load do_scaled * o into registers

    if (dim_idx < dim_head)
        val = do_[dim_idx] * o_[dim_idx];

    // warp shuffle reduce

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (dim_head <= 32) {
        // if dimension of head is 32, no need to reduce across shared memory
        if (dim_idx == 0)
            delta_[seq_idx] = (scalar_t) val;

        return;
    }

    if (lane_id == 0)
        sm_delta[warp_id] = val;

    __syncthreads();

    if (warp_id != 0)
        return;

    // use shared memory to reduce further across warps

    if (dim_idx < (blockDim.x / 32)) {
        val = sm_delta[lane_id];
    } else {
        val = 0.f;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // write out reduced rowsum(do * o)

    if (dim_idx != 0)
        return;

    delta_[seq_idx] = (scalar_t) val;
}

// backward kernel

template <typename scalar_t, typename kv_scalar_t, int tile_size, int dim_head>
__global__ void backward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<float, 3> inv_l,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<scalar_t, 3> attn_bias,
          PackedAccessor<float, 4> dq,
          PackedAccessor<kv_scalar_t, 4> dk,
          PackedAccessor<kv_scalar_t, 4> dv,
          PackedAccessor<float, 3> d_attn_bias,
    const PackedAccessor<scalar_t, 4> d_out_scaled,
    const PackedAccessor<scalar_t, 3> delta,
    const float scale,
    const bool causal,
    const bool has_mask,
    const bool has_attn_bias,
    const bool attn_bias_batch_dim,
    const bool attn_bias_requires_grad,
    const bool is_single_head_kv
) {

    // dimensions

    const int head = q.size(1);

    const int batch_idx = blockIdx.x / head;
    const int head_idx = blockIdx.x % head;
    const int kv_head_idx = is_single_head_kv ? 0 : head_idx;

    const int row_seq_len = q.size(2);
    const int col_seq_len = k.size(2);
    const int seq_len_diff = col_seq_len - row_seq_len;

    constexpr int chunk_size = 16;

    // registers

    using QK_mma_t  = mma::warp_tile<scalar_t, dim_head, tile_size, tile_size>;
    using dV_mma_t = mma::warp_tile<scalar_t, dim_head, tile_size, dim_head>;
    using dK_mma_t = mma::warp_tile<scalar_t, dim_head, tile_size, dim_head>;
    using dQ_mma_t = mma::warp_tile<scalar_t, dim_head, tile_size, dim_head>;

    QK_mma_t QK_mma;
    dV_mma_t dv_mma;
    dK_mma_t dk_mma;
    dQ_mma_t dq_mma;

    // shared memory

    using Q_sm_t_ = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using Q_sm_ = mem::shared_fragment<scalar_t, chunk_size, dim_head>;

    using L_sm_t = mem::shared_fragment<float, 1, tile_size>;
    using D_sm_t = mem::shared_fragment<scalar_t, 1, tile_size>;

    using K_sm_t_ = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using K_sm_ = mem::shared_fragment<scalar_t, chunk_size, dim_head>;
    using V_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;

    using DO_sm_ = mem::shared_fragment<scalar_t, chunk_size, dim_head>;
    using DO_sm_t_ = mem::shared_fragment<scalar_t, chunk_size, tile_size>;

    using C_sm_t = mem::shared_fragment<scalar_t, tile_size, tile_size>;
    using mask_sm_t = mem::shared_fragment<bool, 1, tile_size>;

    using DK_sm_t = mem::shared_fragment<scalar_t, tile_size, dim_head>;
    using DV_sm_t = mem::shared_fragment<scalar_t, tile_size, dim_head>;

    __shared__ scalar_t _shared_mem[layout::smem<scalar_t, tile_size, dim_head>::backward_size];

    auto __shared_mem = reinterpret_cast<char*>(_shared_mem);

    Q_sm_ Q_sm{__shared_mem};
    Q_sm_t_ Q_sm_t{__shared_mem};

    DO_sm_ DO_sm{__shared_mem};
    DO_sm_t_ DO_sm_t{__shared_mem};

    L_sm_t L_sm{__shared_mem};

    DK_sm_t DK_sm{__shared_mem};
    DV_sm_t DV_sm{__shared_mem};

    auto next_ptr = (dim_head > tile_size) ? DO_sm.next() : DO_sm_t.next();

    K_sm_ K_sm{next_ptr};
    K_sm_t_ K_sm_t{next_ptr};
    V_sm_t V_sm{next_ptr};

    auto next_ptr_ = (dim_head > tile_size) ? K_sm.next() : K_sm_t.next();

    D_sm_t D_sm{next_ptr_};

    C_sm_t C_sm{D_sm.next()};
    mask_sm_t mask_sm{D_sm.next()};

    // shortcut accessors

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][kv_head_idx];
    auto v_ = v[batch_idx][kv_head_idx];
    auto l_ = inv_l[batch_idx][head_idx];
    auto dq_ = dq[batch_idx][head_idx];
    auto dk_ = dk[batch_idx][kv_head_idx];
    auto dv_ = dv[batch_idx][kv_head_idx];
    auto delta_ = delta[batch_idx][head_idx];
    auto do_ = d_out_scaled[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // handle attention bias

    auto ds_ = has_attn_bias ? d_attn_bias[attn_bias_batch_dim ? batch_idx : head_idx] : d_attn_bias[0];
    auto bias_ = has_attn_bias ? attn_bias[attn_bias_batch_dim ? batch_idx : head_idx] : attn_bias[0];

    // variables

    float bias;

    // tiles

    auto col_tile_size = tile_size;
    auto row_tile_size = tile_size;

    // loop over column tiles

    int col_tile_offset = (blockIdx.y * col_tile_size);

    dv_mma.zero();
    dk_mma.zero();

    for (int row_tile_offset = 0; row_tile_offset < row_seq_len; row_tile_offset += row_tile_size) {

        if (causal && ((col_tile_offset - seq_len_diff) >= (row_tile_offset + row_tile_size)))
            continue;

        QK_mma.zero();

        // get qk similarity matrix

        for (int k = 0; k < dim_head; k += chunk_size) {
            Q_sm_t.load_transpose(q_, k, row_tile_offset, 0, row_seq_len);
            K_sm_t.load_transpose(k_, k, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(Q_sm_t, K_sm_t, 0, 0, chunk_size);
            __syncthreads();
        }

        // load rowsums and mask if needed

        L_sm.store_row(l_, row_tile_offset, row_tile_size, row_seq_len);

        if (has_mask)
            mask_sm.store_row(mask_, col_tile_offset, col_tile_size, col_seq_len);

        __syncthreads();

        // scale and exponentiation, depending on masking or not

        QK_mma.pointwise([&](scalar_t el, int col, int row) -> scalar_t {
            int attn_col = col + col_tile_offset;
            int attn_row = row + row_tile_offset;

            if ((attn_col >= col_seq_len) ||
                (attn_row >= row_seq_len) ||
                (causal && ((attn_col - seq_len_diff) > attn_row)) ||
                (has_mask && !mask_sm.smem[col]))
                return 0.f;

            float shift = scale;

            if (causal && seq_len_diff == 0)
                if (attn_row == 0)
                    return 1.f;

                shift = min(shift, (float) attn_row);

            bias = has_attn_bias ? (float) bias_[attn_row][attn_col] : 0.f;

            return __expf(scale * el + bias - shift) * L_sm.smem[row];
        });

        if (has_mask)
            __syncthreads();

        QK_mma.store(C_sm);

        __syncthreads();

        // aggregate dv with recomputed attention matrix

        for (int k = 0; k < row_tile_size; k += chunk_size) {
            DO_sm.load(do_, 0, row_tile_offset + k, 0, row_seq_len);
            __syncthreads();

            dv_mma.mma(C_sm, DO_sm, k, 0, chunk_size);
            __syncthreads();
        }

        // calculate dp

        QK_mma.zero();

        for (int k = 0; k < dim_head; k += chunk_size) {
            DO_sm_t.load_transpose(do_, k, row_tile_offset, 0, row_seq_len);
            V_sm.load_transpose(v_, k, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(DO_sm_t, V_sm, 0, 0, chunk_size);
            __syncthreads();
        }

        // load pre-calculated delta

        __syncthreads();

        D_sm.store_row(delta_, row_tile_offset, row_tile_size, row_seq_len);

        __syncthreads();

        // delta = rowsum(do * o), precomputed in backward preprocess
        // calculate ds = (dp - delta) * p

        QK_mma.pointwise([&](scalar_t el, int col, int row) -> scalar_t {
            return (el - D_sm.smem[row]) * C_sm(col, row);
        });

        __syncthreads();

        // accumulate to ds if needed

        if (has_attn_bias) {
            QK_mma.atomic_add(ds_, row_tile_offset, col_tile_offset, row_seq_len, col_seq_len);
        }

        // scale

        QK_mma.pointwise([&](scalar_t el, int, int) -> scalar_t {
            return el * scale;
        });

        __syncthreads();

        QK_mma.store(C_sm);

        __syncthreads();

        // calculate dk

        for (int k = 0; k < row_tile_size; k += chunk_size) {
            Q_sm.load(q_, 0, row_tile_offset + k, 0, row_seq_len);
            __syncthreads();

            dk_mma.mma(C_sm, Q_sm, k, 0, chunk_size);
            __syncthreads();
        }

        QK_mma.store_transpose(C_sm);

        __syncthreads();

        dq_mma.zero();

        for (int k = 0; k < col_tile_size; k += chunk_size) {
            K_sm.load(k_, 0, col_tile_offset + k, 0, col_seq_len);
            __syncthreads();

            dq_mma.mma(C_sm, K_sm, k, 0, chunk_size);
            __syncthreads();
        }

        dq_mma.atomic_add(dq_, row_tile_offset, 0, row_seq_len, dim_head);
    }

    if (is_single_head_kv) {
        dv_mma.atomic_add(dv_, col_tile_offset, 0, col_seq_len, dim_head);

        dk_mma.atomic_add(dk_, col_tile_offset, 0, col_seq_len, dim_head);

    } else {
        dv_mma.store(dv_, DV_sm, 0, col_tile_offset, 0, col_seq_len);

        __syncthreads();

        dk_mma.store(dk_, DK_sm, 0, col_tile_offset, 0, col_seq_len);
    }
}

// forwards c++ function

std::tuple<at::Tensor, at::Tensor, bool> flash_cosine_sim_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    at::optional<torch::Tensor> mask,
    at::optional<torch::Tensor> attn_bias,
    bool attn_bias_batch_dim,
    float scale,
    bool causal
) {
    auto q_scalar_type = q.scalar_type();
    auto query_device = device_of(q);

    const at::cuda::OptionalCUDAGuard device_guard(query_device);

    // single headed key / values

    bool is_merged_batch_head = q.ndimension() == 3;

    if (is_merged_batch_head) {
        assert(('if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions', k.ndimension() == 3 && v.ndimension() == 3));

        attn_bias_batch_dim = true;
        q = q.unsqueeze(1);
    }

    if (k.ndimension() == 3)
        k = k.unsqueeze(1);

    if (v.ndimension() == 3)
        v = v.unsqueeze(1);

    // dimensions

    const int batch     = q.size(0);
    const int heads     = q.size(1);
    const int kv_heads  = k.size(1);
    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int q_dim     = q.size(3);
    const int k_dim     = k.size(3);
    const int v_dim     = v.size(3);

    assert(("query, key, value dimensions must be the same", q_dim == k_dim && k_dim == v_dim));
    assert(("only dimensions 32, 64, 128 allowed for now", q_dim == 32 || q_dim == 64 || q_dim == 96 || q_dim == 128));
    assert(("mask should not be given if causal", !(causal && mask.has_value())));

    // derived values

    const bool is_single_head_kv = heads > 1 && kv_heads == 1;

    const bool has_attn_bias     = attn_bias.has_value();
    const bool has_mask          = mask.has_value();

    auto options = torch::TensorOptions().device(query_device);

    // optionals

    auto mask_value = has_mask ? mask.value() : at::empty({batch, 0}, options.dtype(torch::kBool));
    auto attn_bias_value = has_attn_bias ? attn_bias.value() : at::empty({1, 0, 0}, options.dtype(q_scalar_type));

    // should backwards, determines whether to store row sum

    bool should_backwards = q.requires_grad() || k.requires_grad() || v.requires_grad() || attn_bias_value.requires_grad();

    // create intermediate or output tensors

    auto o = at::empty({batch, heads, q_seq_len, v_dim}, options.dtype(q_scalar_type));
    auto l = at::empty({batch, heads, should_backwards ? q_seq_len : 0}, options.dtype(at::kFloat));

    // dispatch forward call

    AT_TYPE_DISPATCH_SWITCH(q_scalar_type, scalar_t, (at::ScalarType::Float, at::ScalarType::Half, at::ScalarType::BFloat16), (
        VALUE_DISPATCH_SWITCH(v_dim, dim_head, (32, 64, 96, 128), (

            const int tile_size = 64;

            const dim3 threads_per_block(layout::tpb<scalar_t, dim_head>::TPB);

            const dim3 blocks(
                cdiv(q_seq_len, tile_size),
                batch,
                heads
            );

            forward_kernel<scalar_t, tile_size, dim_head><<<blocks, threads_per_block>>>(
                ACCESSOR(q, 4, scalar_t),
                ACCESSOR(k, 4, scalar_t),
                ACCESSOR(v, 4, scalar_t),
                ACCESSOR(o, 4, scalar_t),
                ACCESSOR(l, 3, float),
                ACCESSOR(mask_value, 2, bool),
                ACCESSOR(attn_bias_value, 3, scalar_t),
                scale,
                causal,
                has_mask,
                has_attn_bias,
                attn_bias_batch_dim,
                should_backwards,
                is_single_head_kv
            );

        ), ())
    ), ())

    if (is_merged_batch_head)
        o = o.squeeze(1);

    // handle error    

    CHECK_LAST_CUDA_ERROR(true);

    return { o, l, should_backwards};
}

// backwards c++ function

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, at::optional<torch::Tensor>> flash_cosine_sim_attention_backward(
    torch::Tensor d_out,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    at::optional<torch::Tensor> mask,
    at::optional<torch::Tensor> attn_bias,
    bool attn_bias_batch_dim,
    float scale,
    bool causal
) {
    auto q_scalar_type = q.scalar_type();
    auto query_device = device_of(q);

    const at::cuda::OptionalCUDAGuard device_guard(query_device);

    // single headed key / values

    bool q_no_heads = q.ndimension() == 3;
    bool k_no_heads = k.ndimension() == 3;
    bool v_no_heads = v.ndimension() == 3;

    if (q_no_heads)
        q = q.unsqueeze(1);

    if (o.ndimension() == 3)
        o = o.unsqueeze(1);

    if (d_out.ndimension() == 3)
        d_out = d_out.unsqueeze(1);

    if (k_no_heads)
        k = k.unsqueeze(1);

    if (v_no_heads)
        v = v.unsqueeze(1);

    // dimensions and derived values

    const int batch    = q.size(0);
    const int heads    = q.size(1);
    const int kv_heads = k.size(1);
    const int seq      = q.size(2);
    const int k_seq    = k.size(2);
    const int k_dim    = k.size(3);
    const int v_dim    = v.size(3);

    const bool is_single_head_kv = heads > 1 && kv_heads == 1;

    const bool has_attn_bias     = attn_bias.has_value();
    const bool has_mask          = mask.has_value();

    const bool attn_bias_requires_grad = has_attn_bias && attn_bias.value().requires_grad();

    auto options = torch::TensorOptions().device(query_device);

    // optionals

    auto mask_value = has_mask ? mask.value() : at::empty({batch, 0}, options.dtype(torch::kBool));
    auto attn_bias_value = has_attn_bias ? attn_bias.value() : at::empty({1, 0, 0}, options.dtype(q_scalar_type));

    // create intermediate or output tensors

    auto options_kfloat_dtype = options.dtype(torch::kFloat);
    auto options_q_dtype = options.dtype(q_scalar_type);

    auto delta = at::empty({batch, heads, seq}, options.dtype(q_scalar_type));

    auto dq = at::zeros_like(q, options_kfloat_dtype);

    auto dk = is_single_head_kv ? at::zeros_like(k, options_kfloat_dtype) : at::empty_like(k, options_q_dtype);
    auto dv = is_single_head_kv ? at::zeros_like(v, options_kfloat_dtype) : at::empty_like(k, options_q_dtype);

    auto db = (has_attn_bias && attn_bias_requires_grad) ? at::zeros_like(attn_bias_value, options_kfloat_dtype) : at::empty({attn_bias_value.size(0), 0, 0}, options_kfloat_dtype);

    auto dk_scalar_type = dk.scalar_type();

    // setup backwards call

    AT_TYPE_DISPATCH_SWITCH(dk_scalar_type, kv_scalar_t, (at::ScalarType::Float, at::ScalarType::Half), (
        AT_TYPE_DISPATCH_SWITCH(q_scalar_type, scalar_t, (at::ScalarType::Float, at::ScalarType::Half), (
            VALUE_DISPATCH_SWITCH(v_dim, dim_head, (32, 64, 96, 128), (

                const int tile_size = 64;

                const dim3 preprocess_threads_per_block(dim_head);

                const dim3 preprocess_blocks(batch * heads, seq);

                backward_preprocess<scalar_t, dim_head><<<preprocess_blocks, preprocess_threads_per_block>>>(
                    ACCESSOR(d_out, 4, scalar_t),
                    ACCESSOR(o, 4, scalar_t),
                    ACCESSOR(delta, 3, scalar_t)
                );

                const dim3 backwards_threads_per_block(layout::tpb<scalar_t, dim_head>::TPB);

                const dim3 backwards_blocks(
                    batch * heads,
                    cdiv(k_seq, tile_size)
                );

                backward_kernel<scalar_t, kv_scalar_t, tile_size, dim_head><<<backwards_blocks, backwards_threads_per_block>>>(
                    ACCESSOR(q, 4, scalar_t),
                    ACCESSOR(k, 4, scalar_t),
                    ACCESSOR(v, 4, scalar_t),
                    ACCESSOR(l, 3, float),
                    ACCESSOR(mask_value, 2, bool),
                    ACCESSOR(attn_bias_value, 3, scalar_t),
                    ACCESSOR(dq, 4, float),
                    ACCESSOR(dk, 4, kv_scalar_t),
                    ACCESSOR(dv, 4, kv_scalar_t),
                    ACCESSOR(db, 3, float),
                    ACCESSOR(d_out, 4, scalar_t),
                    ACCESSOR(delta, 3, scalar_t),
                    scale,
                    causal,
                    has_mask,
                    has_attn_bias,
                    attn_bias_batch_dim,
                    attn_bias_requires_grad,
                    is_single_head_kv
                );

            ), ())
        ), ())
    ), ())

    // handle error

    CHECK_LAST_CUDA_ERROR(true);

    // deal with single headed kv

    if (q_no_heads)
        dq = dq.squeeze(1);

    if (k_no_heads)
        dk = dk.squeeze(1);

    if (v_no_heads)
        dv = dv.squeeze(1);

    // cast back to original type of queries

    dq = dq.to(q_scalar_type);

    if (dk_scalar_type != q_scalar_type) {
        dk = dk.to(q_scalar_type);
        dv = dv.to(q_scalar_type);
    }

    at::optional<torch::Tensor> return_db;

    if (has_attn_bias)
        return_db = db.to(q_scalar_type);

    return { dq, dk, dv, return_db };
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
