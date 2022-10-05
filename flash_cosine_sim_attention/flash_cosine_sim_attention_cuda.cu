#include <cassert>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// error handler
// from https://leimao.github.io/blog/Proper-CUDA-Error-Checking

#define CHECK_LAST_CUDA_ERROR() check(__FILE__, __LINE__)
void check(const char* file, const int line)
{
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

// Custom dispatch inspired from
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
// https://github.com/swansontec/map-macro

// Macro utilities:

#include <ATen/Dispatch.h>

#define REMOVE_PAREN_IMPL(...) __VA_ARGS__
#define REMOVE_PAREN(args) REMOVE_PAREN_IMPL args

#define EVAL0(...) __VA_ARGS__
#define EVAL1(...) EVAL0(EVAL0(EVAL0(__VA_ARGS__)))
#define EVAL2(...) EVAL1(EVAL1(EVAL1(__VA_ARGS__)))
#define EVAL3(...) EVAL2(EVAL2(EVAL2(__VA_ARGS__)))
#define EVAL4(...) EVAL3(EVAL3(EVAL3(__VA_ARGS__)))
#define EVAL(...)  EVAL4(EVAL4(EVAL4(__VA_ARGS__)))

#define MAP_END(...)
#define MAP_OUT

#define MAP_GET_END2() 0, MAP_END
#define MAP_GET_END1(...) MAP_GET_END2
#define MAP_GET_END(...) MAP_GET_END1
#define MAP_NEXT0(test, next, ...) next MAP_OUT
#define MAP_NEXT1(test, next) MAP_NEXT0(test, next, 0)
#define MAP_NEXT(test, next)  MAP_NEXT1(MAP_GET_END test, next)

#define MAP0(f, TYPE_NAME, CASE_CODE, x, peek, ...) f(TYPE_NAME, CASE_CODE, x) MAP_NEXT(peek, MAP1)(f, TYPE_NAME, CASE_CODE, peek, __VA_ARGS__)
#define MAP1(f, TYPE_NAME, CASE_CODE, x, peek, ...) f(TYPE_NAME, CASE_CODE, x) MAP_NEXT(peek, MAP0)(f, TYPE_NAME, CASE_CODE, peek, __VA_ARGS__)
#define MAP(f, TYPE_NAME, CASE_CODE, ...) EVAL(MAP1(f, TYPE_NAME, CASE_CODE, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Type dispatch

#define AT_TYPE_DISPATCH_CASE(TYPE_NAME, CASE_CODE, x)                            \
    case x: {                                                                     \
        using TYPE_NAME C10_UNUSED_DISPATCH_CUDA_WORKAROUND =                     \
          typename c10::impl::ScalarTypeToCPPType<x>::type;                       \
        REMOVE_PAREN(CASE_CODE)                                                   \
        break;                                                                    \
    }

#define AT_TYPE_DISPATCH_SWITCH(TYPE, TYPE_NAME, TYPES, CASE_CODE, DEFAULT_CODE)  \
  {                                                                               \
    switch (TYPE) {                                                               \
        MAP(AT_TYPE_DISPATCH_CASE, TYPE_NAME, CASE_CODE, REMOVE_PAREN(TYPES))     \
        default: {                                                                \
            REMOVE_PAREN(DEFAULT_CODE)                                            \
        }                                                                         \
    }                                                                             \
  }

// Value dispatch

#define VALUE_DISPATCH_CASE(VALUE_NAME, CASE_CODE, x)                             \
    case x: {                                                                     \
        constexpr const auto VALUE_NAME = x;                                      \
        REMOVE_PAREN(CASE_CODE)                                                   \
        break;                                                                    \
    }

#define VALUE_DISPATCH_SWITCH(VALUE, VALUE_NAME, VALUES, CASE_CODE, DEFAULT_CODE) \
  {                                                                               \
    switch (VALUE) {                                                              \
        MAP(VALUE_DISPATCH_CASE, VALUE_NAME, CASE_CODE, REMOVE_PAREN(VALUES))     \
        default: {                                                                \
            REMOVE_PAREN(DEFAULT_CODE)                                            \
        }                                                                         \
    }                                                                             \
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

        template<typename accessor, typename F>
        __device__ void store_row(accessor gmem, int row_offset, int row_tile_size, int row_max, F&& op) {
            for (int i = threadIdx.x; i < row_tile_size; i += blockDim.x) {
                int global_row = row_offset + i;

                if (global_row >= row_max)
                    continue;

                smem[i] = op(gmem[global_row]);
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

    __device__ void divide(float* smem, out_warp_tile_t& mma) {
        if (threadIdx.x < N_tile) smem[threadIdx.x] = 1.f / max(acc, 1e-10);

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

// warp tile, done by @ahennequ Arthur Hennequin

#include <mma.h>

namespace mma {
    template<typename scalar_t>
    struct warp_tile {
        // How much data is processed by a single thread:
        static constexpr int N_thread = 4;
        static constexpr int M_thread = 4;

        // Thread layout within a warp:
        static constexpr int N_warp = 8;
        static constexpr int M_warp = 4;
        static_assert(N_warp * M_warp == 32);

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;
        static_assert(N_block * M_block * N_warp * M_warp == 256); // blockDim.x

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_warp * N_block * N_thread;
        static constexpr int M_tile = M_warp * M_block * M_thread;
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

        // Perform an operation on each element, specified by the given lambda, on C
        template<typename F>
        __device__ void foreach(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int row = i * N_warp + thread_y;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int col = j * M_warp  + thread_x;
                    op(C_frag[i * M_thread + j], col, row);
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
    template<>
    struct warp_tile<c10::Half> {
        // How much data is processed by a single thread:
        static constexpr int N_thread = 2;
        static constexpr int M_thread = 1;

        // Thread layout within a warp:
        static constexpr int N_warp = 16;
        static constexpr int M_warp = 16;

        // Warp layout within a block:
        static constexpr int N_block = 2;
        static constexpr int M_block = 4;

        // Dimensions of the tile, in threads:
        static constexpr int N_tile = N_warp * N_block * N_thread;
        static constexpr int M_tile = M_warp * M_block * M_thread;
        static constexpr int K_tile = 16;

        using output_t = half;

        // Registers:
        wmma::fragment<wmma::accumulator, 16, 16, 16, output_t> C_frag[N_thread * M_thread];

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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> A_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> B_frag;

            for (int k = 0; k < D; k += K_tile) {
                // Load a 1 x M fragment of B from shared memory to registers:
                wmma::load_matrix_sync(B_frag, reinterpret_cast<const half*>(&B_sm(warp_x * M_warp, kb0 + k)), B_sm.stride);

                #pragma unroll
                for (int i = 0; i < N_thread; i++) {
                    // Load a N x 1 fragment of A from shared memory to registers:
                    int y = (warp_y * N_thread + i) * N_warp;
                    wmma::load_matrix_sync(A_frag, reinterpret_cast<const half*>(&A_sm(y, ka0 + k)), A_sm.stride);

                    // Compute:
                    wmma::mma_sync(C_frag[i], A_frag, B_frag, C_frag[i]);
                }
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void pointwise(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    int col = get_warp_col(j) + warp_x * 16;
                    int row = get_warp_row(j) + i * 16 + warp_y * 32;
                    C_frag[i].x[j] = op(C_frag[i].x[j], col, row);
                }
            }
        }

        // Perform a pointwise operation, specified by the given lambda, on C
        template<typename F>
        __device__ void foreach(F&& op) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                #pragma unroll
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    int col = get_warp_col(j) + warp_x * 16;
                    int row = get_warp_row(j) + i * 16 + warp_y * 32;
                    op(C_frag[i].x[j], col, row);
                }
            }
        }

        __device__ int get_warp_row(int i) {
            int tid = threadIdx.x & 31;
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
            int tid = threadIdx.x & 31;
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

        // Copy C from registers to shared memory
        template<typename shared_fragment>
        __device__ void store(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int x = (warp_x * M_thread + j) * M_warp;
                    wmma::store_matrix_sync(reinterpret_cast<half*>(&C_sm(x, y)), C_frag[i * M_thread + j], shared_fragment::stride, wmma::mem_row_major);
                }
            }
        }

        template<typename shared_fragment>
        __device__ void store_transpose(shared_fragment& C_sm) {
            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                int y = (warp_y * N_thread + i) * N_warp;
                #pragma unroll
                for (int j = 0; j < M_thread; j++) {
                    int x = (warp_x * M_thread + j) * M_warp;
                    wmma::store_matrix_sync(reinterpret_cast<half*>(&C_sm(y, x)), C_frag[i * M_thread + j], shared_fragment::stride, wmma::mem_col_major);
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
                for (int j = 0; j < C_frag[i].num_elements; j++) {
                    int col = col_tile_offset + get_warp_col(j) + warp_x * 16;
                    int row = row_tile_offset + get_warp_row(j) + i * 16 + warp_y * 32;

                    if (col >= col_max || row >= row_max)
                        continue;

                    atomicAdd((float*) &gmem[row][col], __half2float(C_frag[i].x[j]));
                }
            }
        }
    };
}

// forward kernel

template <typename scalar_t, int tile_size>
__global__ void forward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
          PackedAccessor<scalar_t, 4> o,
          PackedAccessor<float, 3> l,
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

    const int dim_qk = q.size(3);

    constexpr int chunk_size = 16;

    using QK_mma_t  = mma::warp_tile<scalar_t>;
    using out_mma_t = mma::warp_tile<scalar_t>;

    using Q_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using K_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using V_sm_t = mem::shared_fragment<scalar_t, chunk_size, out_mma_t::M_tile>;
    using C_sm_t = mem::shared_fragment<scalar_t, tile_size, tile_size>;
    using mask_sm_t = mem::shared_fragment<bool, 2, tile_size>;
    using L_sm_t = mem::shared_fragment<float, tile_size, 1>;
    using O_sm_t = mem::shared_fragment<scalar_t, tile_size, tile_size>;

    const int row_tile_offset = blockIdx.x * tile_size;

    // registers

    float bias;

    QK_mma_t  QK_mma;
    out_mma_t out_mma;

    rowsum_accumulator<scalar_t, QK_mma_t, out_mma_t> L_acc;

    // shared memory

    __shared__ scalar_t _shared_mem[Q_sm_t::size + K_sm_t::size + C_sm_t::size];

    auto __shared_mem = reinterpret_cast<char*>(_shared_mem);

    Q_sm_t Q_sm{__shared_mem};
    V_sm_t V_sm{__shared_mem};
    K_sm_t K_sm{Q_sm.next()};
    C_sm_t C_sm{K_sm.next()};
    L_sm_t L_sm{K_sm.next()};
    O_sm_t O_sm{K_sm.next()};
    mask_sm_t mask_sm{K_sm.next()};

    out_mma.zero();
    L_acc.zero();

    // shortcut accessors

    auto Q_ = q[batch][heads];
    auto K_ = k[batch][kv_heads];
    auto V_ = v[batch][kv_heads];
    auto l_ = l[batch][heads];
    auto O_ = o[batch][heads];
    auto mask_ = mask[batch];
    auto bias_ = attn_bias[attn_bias_batch_dim ? batch : heads];

    // renamed vars

    auto col_tile_size = tile_size;
    auto row_tile_size = tile_size;

    // loop over column tiles

    for (int col_tile_offset = 0; col_tile_offset < col_seq_len; col_tile_offset += col_tile_size) {
        if (causal && ((col_tile_offset - seq_len_diff) >= (row_tile_offset + row_tile_size)))
            continue;

        QK_mma.zero();

        // get qk similarity matrix

        for (int i = 0; i < dim_qk; i += chunk_size) {
            Q_sm.load_transpose(Q_, i, row_tile_offset, 0, row_seq_len);
            K_sm.load_transpose(K_, i, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(Q_sm, K_sm, 0, 0, chunk_size);
            __syncthreads();
        }

        // store mask into smem if needed

        if (has_mask)
            mask_sm.store_row(mask_, col_tile_offset, col_tile_size, col_seq_len, [](bool el) {return el;});

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

            if (causal && seq_len_diff == 0 && attn_row == 0 && attn_col == 0)
                return 1.f;

            bias = has_attn_bias ? (float) bias_[attn_row][attn_col] : 0.f;

            return __expf(scale * el + bias - scale);
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

    if (need_store_rowsum)
        L_acc.store(l_, row_tile_offset, row_seq_len);

    L_acc.divide(L_sm.smem, out_mma);

    out_mma.store(O_, O_sm, 0, row_tile_offset, 0, row_seq_len);
}

// backward kernel

// backwards preprocess

// delta = rowsum(do * o)

// done by @ptillet at https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

template <typename scalar_t, int dim_head>
__global__ void backward_preprocess(
    const PackedAccessor<scalar_t, 4> d_out,
    const PackedAccessor<scalar_t, 4> o,
          PackedAccessor<float, 3> delta
) {
    const int heads = o.size(1);
    const int v_dim = o.size(3);

    const int batch_idx = blockIdx.x / heads;
    const int head_idx = blockIdx.x % heads;
    const int seq_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;

    const unsigned mask = __ballot_sync(0xFFFFFFFFU, dim_idx < v_dim);

    // registers

    float val = 0.0f;

    // shared memory

    __shared__ float _shared_mem_preprocess[dim_head / 32];

    float* sm_delta  = reinterpret_cast<float*>(&_shared_mem_preprocess);

    // global mem accessors

    auto do_ = d_out[batch_idx][head_idx][seq_idx];
    auto o_ = o[batch_idx][head_idx][seq_idx];
    auto delta_ = delta[batch_idx][head_idx];

    // load do_scaled * o into registers

    if (dim_idx < v_dim)
        val = do_[dim_idx] * o_[dim_idx]; // todo: do the trick where one reduction step is taken

    // warp shuffle reduce

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
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

// main backward kernel

template <typename scalar_t, int tile_size>
__global__ void backward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<float, 3> l,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<scalar_t, 3> attn_bias,
          PackedAccessor<float, 4> dq,
          PackedAccessor<float, 4> dk,
          PackedAccessor<float, 4> dv,
          PackedAccessor<float, 3> d_attn_bias,
    const PackedAccessor<scalar_t, 4> d_out_scaled,
    const PackedAccessor<float, 3> delta,
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

    const int dim_qk = k.size(3);
    const int dim_v = v.size(3);

    constexpr int chunk_size = 16;

    // registers

    using QK_mma_t  = mma::warp_tile<scalar_t>;
    using dV_mma_t = mma::warp_tile<scalar_t>;
    using dK_mma_t = mma::warp_tile<scalar_t>;
    using dQ_mma_t = mma::warp_tile<scalar_t>;

    QK_mma_t QK_mma;
    dV_mma_t dv_mma;
    dK_mma_t dk_mma;
    dQ_mma_t dq_mma;

    // shared memory

    using Q_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using L_sm_t = mem::shared_fragment<float, 1, tile_size>;
    using DO_sm_t = mem::shared_fragment<scalar_t, chunk_size, dV_mma_t::M_tile>;
    using D_sm_t = mem::shared_fragment<scalar_t, 1, tile_size>;

    using K_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;
    using V_sm_t = mem::shared_fragment<scalar_t, chunk_size, tile_size>;

    using C_sm_t = mem::shared_fragment<scalar_t, tile_size, tile_size>;
    using DK_sm_t = mem::shared_fragment<scalar_t, tile_size, dK_mma_t::M_tile>;
    using DV_sm_t = mem::shared_fragment<scalar_t, tile_size, dV_mma_t::M_tile>;

    __shared__ scalar_t _shared_mem[Q_sm_t::size + K_sm_t::size + C_sm_t::size + D_sm_t::size];

    auto __shared_mem = reinterpret_cast<char*>(_shared_mem);

    Q_sm_t Q_sm{__shared_mem};
    DO_sm_t DO_sm{__shared_mem};
    L_sm_t L_sm{__shared_mem};

    K_sm_t K_sm{Q_sm.next()};
    V_sm_t V_sm{Q_sm.next()};
    D_sm_t D_sm{K_sm.next()};
    C_sm_t C_sm{D_sm.next()};
    DK_sm_t DK_sm{D_sm.next()};
    DV_sm_t DV_sm{D_sm.next()};

    // shortcut accessors

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][kv_head_idx];
    auto v_ = v[batch_idx][kv_head_idx];
    auto l_ = l[batch_idx][head_idx];
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

        for (int k = 0; k < dim_qk; k += chunk_size) {
            Q_sm.load_transpose(q_, k, row_tile_offset, 0, row_seq_len);
            K_sm.load_transpose(k_, k, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(Q_sm, K_sm, 0, 0, chunk_size);
            __syncthreads();
        }

        // load rowsums

        L_sm.store_row(l_, row_tile_offset, row_tile_size, row_seq_len, [&](float el) {
            return 1.f / max(el, 1e-10);
        });

        __syncthreads();

        // scale and exponentiation, depending on masking or not

        QK_mma.pointwise([&](scalar_t el, int col, int row) -> scalar_t {
            int attn_col = col + col_tile_offset;
            int attn_row = row + row_tile_offset;

            if ((attn_col >= col_seq_len) ||
                (attn_row >= row_seq_len) ||
                (causal && ((attn_col - seq_len_diff) > attn_row)) ||
                (has_mask && !mask_[attn_col]))
                return 0.f;

            if (causal && seq_len_diff == 0 && attn_row == 0 && attn_col == 0)
                return 1.f;

            bias = has_attn_bias ? (float) bias_[attn_row][attn_col] : 0.f;

            return __expf(scale * el + bias - scale) * L_sm.smem[row];
        });

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

        for (int k = 0; k < dim_v; k += chunk_size) {
            DO_sm.load_transpose(do_, k, row_tile_offset, 0, row_seq_len);
            V_sm.load_transpose(v_, k, col_tile_offset, 0, col_seq_len);
            __syncthreads();

            QK_mma.mma(DO_sm, V_sm, 0, 0, chunk_size);
            __syncthreads();
        }

        // load pre-calculated delta

        __syncthreads();

        D_sm.store_row(delta_, row_tile_offset, row_tile_size, row_seq_len, [](scalar_t el) {return el;});

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

        dq_mma.atomic_add(dq_, row_tile_offset, 0, row_seq_len, dim_qk);
    }

    if (is_single_head_kv) {
        dv_mma.atomic_add(dv_, col_tile_offset, 0, col_seq_len, dim_v);

        dk_mma.atomic_add(dk_, col_tile_offset, 0, col_seq_len, dim_qk);

    } else {
        dv_mma.store(dv_, DV_sm, 0, col_tile_offset, 0, col_seq_len);

        __syncthreads();

        dk_mma.store(dk_, DK_sm, 0, col_tile_offset, 0, col_seq_len);
    }
}

// forwards c++ function

std::vector<at::Tensor> flash_cosine_sim_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    bool attn_bias_batch_dim,
    float scale,
    bool causal,
    bool need_store_rowsum
) {
    auto q_scalar_type = q.scalar_type();
    auto query_device = device_of(q);

    const at::cuda::OptionalCUDAGuard device_guard(query_device);

    const int batch     = q.size(0);
    const int heads     = q.size(1);
    const int kv_heads  = k.size(1);
    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim     = q.size(3);
    const int v_dim     = v.size(3);

    const bool is_single_head_kv = heads > 1 && kv_heads == 1;
    const bool has_attn_bias     = !!attn_bias.numel();
    const bool has_mask          = !!mask.numel();

    // create intermediate or output tensors

    auto options = torch::TensorOptions().device(query_device);

    auto o = at::empty({batch, heads, q_seq_len, v_dim}, options.dtype(q_scalar_type));
    auto l = at::empty({batch, heads, need_store_rowsum ? q_seq_len : 0}, options.dtype(at::kFloat));

    // setup threads per block

    const int tile_size = 64;

    // dispatch forward call

    AT_TYPE_DISPATCH_SWITCH(q_scalar_type, scalar_t, (at::ScalarType::Float, at::ScalarType::Half), (
        VALUE_DISPATCH_SWITCH(v_dim, dim_head, (64), (

            const dim3 threads_per_block(256);

            const dim3 blocks(
                cdiv(q_seq_len, tile_size),
                batch,
                heads
            );

            forward_kernel<scalar_t, tile_size><<<blocks, threads_per_block>>>(
                ACCESSOR(q, 4, scalar_t),
                ACCESSOR(k, 4, scalar_t),
                ACCESSOR(v, 4, scalar_t),
                ACCESSOR(o, 4, scalar_t),
                ACCESSOR(l, 3, float),
                ACCESSOR(mask, 2, bool),
                ACCESSOR(attn_bias, 3, scalar_t),
                scale,
                causal,
                has_mask,
                has_attn_bias,
                attn_bias_batch_dim,
                need_store_rowsum,
                is_single_head_kv
            );

        ), ())
    ), ())

    // handle error

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();

    return { o, l };
}

// backwards c++ function

std::vector<torch::Tensor> flash_cosine_sim_attention_backward(
    torch::Tensor d_out,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    bool attn_bias_batch_dim,
    float scale,
    bool causal,
    bool attn_bias_requires_grad
) {
    auto q_scalar_type = q.scalar_type();
    auto query_device = device_of(q);

    const at::cuda::OptionalCUDAGuard device_guard(query_device);

    const int batch    = q.size(0);
    const int heads    = q.size(1);
    const int kv_heads = k.size(1);
    const int seq      = q.size(2);
    const int k_seq    = k.size(2);
    const int k_dim    = k.size(3);
    const int v_dim    = v.size(3);

    const bool is_single_head_kv = heads > 1 && kv_heads == 1;
    const bool has_attn_bias     = !!attn_bias.numel();
    const bool has_mask          = !!mask.numel();

    // create intermediate or output tensors

    auto options = torch::TensorOptions().device(query_device);

    auto delta = at::empty({batch, heads, seq}, options.dtype(torch::kFloat));

    auto dq = at::zeros_like(q, options.dtype(torch::kFloat));
    auto dk = at::zeros_like(k, options.dtype(torch::kFloat));
    auto dv = at::zeros_like(v, options.dtype(torch::kFloat));

    auto db = (has_attn_bias && attn_bias_requires_grad) ? at::zeros_like(attn_bias, options.dtype(torch::kFloat)) : at::empty({attn_bias.size(0), 0, 0}, options.dtype(torch::kFloat));

    // setup threads per block

    const int tile_size = 64;

    // setup backwards call

    AT_TYPE_DISPATCH_SWITCH(q_scalar_type, scalar_t, (at::ScalarType::Float, at::ScalarType::Half), (
        VALUE_DISPATCH_SWITCH(v_dim, dim_head, (64), (

            const dim3 backwards_preprocess_threads_per_block(dim_head);
            const dim3 backwards_threads_per_block(256);

            const dim3 backwards_blocks(
                batch * heads,
                cdiv(k_seq, tile_size)
            );

            const dim3 backwards_preprocess_blocks(batch * heads, seq);

            backward_preprocess<scalar_t, dim_head><<<backwards_preprocess_blocks, backwards_preprocess_threads_per_block>>>(
                ACCESSOR(d_out, 4, scalar_t),
                ACCESSOR(o, 4, scalar_t),
                ACCESSOR(delta, 3, float)
            );

            backward_kernel<scalar_t, tile_size><<<backwards_blocks, backwards_threads_per_block>>>(
                ACCESSOR(q, 4, scalar_t),
                ACCESSOR(k, 4, scalar_t),
                ACCESSOR(v, 4, scalar_t),
                ACCESSOR(l, 3, float),
                ACCESSOR(mask, 2, bool),
                ACCESSOR(attn_bias, 3, scalar_t),
                ACCESSOR(dq, 4, float),
                ACCESSOR(dk, 4, float),
                ACCESSOR(dv, 4, float),
                ACCESSOR(db, 3, float),
                ACCESSOR(d_out, 4, scalar_t),
                ACCESSOR(delta, 3, float),
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

    cudaDeviceSynchronize();

    // handle error

    CHECK_LAST_CUDA_ERROR();

    return { dq, dk, dv, db };
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
