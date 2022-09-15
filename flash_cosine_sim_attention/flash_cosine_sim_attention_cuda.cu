#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
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

__host__ __device__ int next_multiple_of(int num, int multiple_of) {
    return cdiv(num, multiple_of) * multiple_of;
}

__host__ __device__ int next_pow_2(int n) {
    int i = 1;
    while(i < n)
        i *= 2;
    return i;
}

__device__ void warp_reduce(volatile float* sm, int tid, int max) {
    for (int s = 32; s > 0; s>>=1) {
        if ((tid + s) >= max)
            continue;

        sm[tid] += sm[tid + s];
    }
}

bool divisible_by(int num, int denom) {
    return (num % denom) == 0;
}

// mma and smem fragment

// mma

struct mma_warp_tile {
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

    static constexpr float IS_NULL_FLOAT = -3.14159e6;
    static constexpr float MASK_VALUE = -1e8;

    // Registers:
    float A_frag[N_thread];            // N x 1 fragment
    float B_frag[M_thread];            // 1 x M fragment
    float C_frag[N_thread * M_thread]; // N x M fragment


    int warp_x;   // x offset of the warp within the block tile
    int warp_y;   // y offset of the warp within the block tile
    int thread_x; // x offset of the thread within the warp tile
    int thread_y; // y offset of the thread within the warp tile

    __device__ mma_warp_tile() {
        int warp_id = threadIdx.x / 32;
        warp_x = (warp_id % M_block);
        warp_y = (warp_id / M_block);

        int lane_id = threadIdx.x % 32;
        thread_x = warp_x * M_warp * M_thread + lane_id % M_warp;
        thread_y = warp_y * N_warp * N_thread + lane_id / M_warp;
    }

    // Initialize C to all zeros
    __device__ void zero() {
        for (int i = 0; i < N_thread * M_thread; i++) {
            C_frag[i] = 0.f;
        }
    }

    // Performs C = A * B + C
    __device__ void mma(
        const float* A_sm_ptr,
        const float* B_sm_ptr,
        int k,
        bool has_mask,
        const float is_null_float
    ) {
        // Load a N x 1 fragment of A from shared memory to registers:
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            A_frag[i] = A_sm_ptr[i * N_warp + thread_y + k * N_tile];
        }

        // Load a 1 x M fragment of B from shared memory to registers:
        #pragma unroll
        for (int i = 0; i < M_thread; i++) {
            B_frag[i] = B_sm_ptr[i * M_warp + thread_x + k * M_tile];
        }

        // Compute:
        #pragma unroll
        for (int j = 0; j < M_thread ; j++) {

            bool is_masked_out = false;
            if (has_mask) {
                is_masked_out = B_sm_ptr[j * M_warp + thread_x] == is_null_float;
            }

            #pragma unroll
            for (int i = 0; i < N_thread; i++) {
                if (is_masked_out) {
                    C_frag[i * M_thread + j] = MASK_VALUE;
                } else {
                    C_frag[i * M_thread + j] += A_frag[i] * B_frag[j];
                }
            }
        }
    }

    // Perform a pointwise operation, specified by the given lambda, on C
    template<typename F>
    __device__ void pointwise(F&& op) {
        #pragma unroll
        for (int i = 0; i < N_thread * M_thread; i++) {
            C_frag[i] = op(C_frag[i], i);
        }
    }

    // Copy C from registers to shared memory
    __device__ void store(float* C_sm_ptr) {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            #pragma unroll
            for (int j = 0; j < M_thread ; j++) {
                C_sm_ptr[(thread_y + i * N_warp) * M_tile + j * M_warp + thread_x]
                  = C_frag[i * M_thread + j];
            }
        }
    }

    __device__ void store_transpose(float* C_sm_ptr) {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            #pragma unroll
            for (int j = 0; j < M_thread ; j++) {
                C_sm_ptr[thread_y + i * N_warp + (j * M_warp + thread_x) * N_tile]
                  = C_frag[i * M_thread + j];
            }
        }
    }
};


struct out_mma_warp_tile {
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

    static constexpr float EPS = 1e-10;

    // Registers:
    float A_frag[N_thread];            // N x 1 fragment
    float B_frag[M_thread];            // 1 x M fragment
    float L_frag[N_thread];            // N x 1 fragment
    float C_frag[N_thread * M_thread]; // N x M fragment

    int warp_x;   // x offset of the warp within the block tile
    int warp_y;   // y offset of the warp within the block tile
    int thread_x; // x offset of the thread within the warp tile
    int thread_y; // y offset of the thread within the warp tile

    __device__ out_mma_warp_tile() {
        int warp_id = threadIdx.x / 32;
        warp_x = (warp_id % M_block);
        warp_y = (warp_id / M_block);

        int lane_id = threadIdx.x % 32;
        thread_x = warp_x * M_warp * M_thread + lane_id % M_warp;
        thread_y = warp_y * N_warp * N_thread + lane_id / M_warp;
    }

    // Initialize C to all zeros
    __device__ void zero() {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            L_frag[i] = 0.f;
        }

        #pragma unroll
        for (int i = 0; i < N_thread * M_thread; i++) {
            C_frag[i] = 0.f;
        }
    }

    // Performs C = A * B + C
    __device__ void mma(
        const float* A_sm_ptr,
        const float* B_sm_ptr,
        int k
    ) {
        // Load a N x 1 fragment of A from shared memory to registers:
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            A_frag[i] = A_sm_ptr[i * N_warp + thread_y + k * N_tile];
        }

        // Load a 1 x M fragment of B from shared memory to registers:
        #pragma unroll
        for (int i = 0; i < M_thread; i++) {
            B_frag[i] = B_sm_ptr[i * M_warp + thread_x + k * M_tile];
        }

        // Compute:
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            L_frag[i] += A_frag[i];

            #pragma unroll
            for (int j = 0; j < M_thread ; j++) {
                C_frag[i * M_thread + j] += A_frag[i] * B_frag[j];
            }
        }
    }

    // Perform a pointwise operation, specified by the given lambda, on C
    template<typename F>
    __device__ void pointwise(F&& op) {
        #pragma unroll
        for (int i = 0; i < N_thread * M_thread; i++) {
            C_frag[i] = op(C_frag[i], i);
        }
    }

    // Copy C from registers to shared memory
    __device__ void store(float* C_sm_ptr) {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            #pragma unroll
            for (int j = 0; j < M_thread ; j++) {
                C_sm_ptr[(thread_y + i * N_warp) * M_tile + j * M_warp + thread_x]
                  = C_frag[i * M_thread + j] / max(L_frag[i], EPS);
            }
        }
    }

    template<typename accessor>
    __device__ void store_rowsum(accessor gmem, int tile_y, int max_y) {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            int gmem_idx = tile_y * N_tile + i * N_warp + thread_y;

            if (gmem_idx >= max_y)
                continue;

            gmem[gmem_idx] = L_frag[i];
        }
    }

    __device__ void store_transpose(float* C_sm_ptr) {
        #pragma unroll
        for (int i = 0; i < N_thread; i++) {
            #pragma unroll
            for (int j = 0; j < M_thread ; j++) {
                C_sm_ptr[thread_y + i * N_warp + (j * M_warp + thread_x) * N_tile]
                  = C_frag[i * M_thread + j] / max(L_frag[i], EPS);
            }
        }
    }
};
// shared memory fragment

template<typename T>
struct smem_fragment {
    T* smem;
    int N;
    int M;

    __device__ smem_fragment(T* shared_base, int N, int M)
      : smem(shared_base), N(N), M(M) { }

    __device__ void load(const T* gmem) {
        for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
            smem[i] = gmem[i];
        }
    }

    template<typename accessor>
    __device__ void load(accessor gmem, int tile_x, int tile_y, int max_y) {
        for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
            int x = i % M;
            int y = i / M;
            int gmem_y = y + tile_y * N;
            int gmem_x = x + tile_x * M;

            if (gmem_y >= max_y)
                continue;

            smem[i] = gmem[gmem_y][gmem_x];
        }
    }

    template<typename accessor>
    __device__ void load_transpose(accessor gmem, int tile_x, int tile_y, int max_y) {
        for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
            int y = i % M;
            int x = i / M;
            int gmem_y = x + tile_y * N;
            int gmem_x = y + tile_x * M;

            if (gmem_y >= max_y)
                continue;

            smem[y * N + x] = gmem[gmem_y][gmem_x];
        }
    }

    template<typename accessor, typename accessor_mask>
    __device__ void load_transpose(accessor gmem, int tile_x, int tile_y, bool has_mask, accessor_mask mask, const float is_null_float, int max_y) {
        if (!has_mask)
            return load_transpose(gmem, tile_x, tile_y, max_y);

        for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
            int y = i % M;
            int x = i / M;
            int gmem_y = x + tile_y * N;
            int gmem_x = y + tile_x * M;

            if (y == 0 && !mask[gmem_y]) {
                smem[y * N + x] = is_null_float;
                continue;
            }

            if (gmem_y >= max_y)
                continue;

            smem[y * N + x] = gmem[gmem_y][gmem_x];
        }
    }

    template<typename accessor>
    __device__ void store(accessor gmem, int tile_x, int tile_y, int max_y) {
        for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
            int x = i % M;
            int y = i / M;
            int gmem_y = y + tile_y * N;
            int gmem_x = x + tile_x * M;

            if (gmem_y >= max_y) {
                continue;
            }

            gmem[gmem_y][gmem_x] = smem[i];
        }
    }

    __device__ unsigned size() {
        return N * M;
    }

    __device__ T* next() {
        return smem + size();
    }
};

// forward kernel

__global__ void forward_kernel(
    const PackedAccessor<float, 4> Q,
    const PackedAccessor<float, 4> K,
    const PackedAccessor<float, 4> V,
          PackedAccessor<float, 4> O,
          PackedAccessor<float, 3> L,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<float, 3> attn_bias,
    const float scale,
    const bool causal,
    const bool has_mask,
    const bool has_attn_bias
) {
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int M = K.size(2);
    const int D = Q.size(3);
    const int E = V.size(3);

    const int MN_DIFF = M - N;  // for calculating causality when query and key lengths differ

    const int batch = blockIdx.y / H;
    const int heads = blockIdx.y % H;

    // shortcut accessor

    auto Q_ = Q[batch][heads];
    auto K_ = K[batch][heads];
    auto V_ = V[batch][heads];
    auto O_ = O[batch][heads];
    auto L_ = L[batch][heads];
    auto attn_bias_ = attn_bias[heads];

    // tiles

    const int tile_w = cdiv(M, mma_warp_tile::M_tile);
    const int tile_y = blockIdx.x;

    // shared memory

    extern __shared__ float _shared_mem[];

    mma_warp_tile QK_mma; // 32x16 tile per warp in registers -> process 64x64 with the block
    out_mma_warp_tile out_mma;

    smem_fragment<float> Q_sm{_shared_mem, mma_warp_tile::N_tile, D};
    smem_fragment<float> O_sm{_shared_mem, mma_warp_tile::N_tile, E};
    smem_fragment<float> A_sm{(E > D ? O_sm.next() : A_sm.next()), mma_warp_tile::N_tile, mma_warp_tile::M_tile};
    smem_fragment<float> K_sm{A_sm.next(), mma_warp_tile::M_tile, D};
    smem_fragment<float> V_sm{A_sm.next(), mma_warp_tile::M_tile, E};

    // helper variables

    int global_row, global_col;
    float bias;

    // start loop

    out_mma.zero();

    Q_sm.load_transpose(Q_, 0, tile_y, N);

    for (int tile_x = 0; tile_x < tile_w; tile_x++) {
        if (causal && (mma_warp_tile::M_tile * tile_x - MN_DIFF) >= (mma_warp_tile::N_tile * (tile_y + 1)))
            continue;

        K_sm.load_transpose(K_, 0, tile_x, has_mask, mask[batch], mma_warp_tile::IS_NULL_FLOAT, M);

        __syncthreads();

        QK_mma.zero();

        for (int d = 0; d < D; d++) {
            QK_mma.mma(Q_sm.smem, K_sm.smem, d, has_mask, mma_warp_tile::IS_NULL_FLOAT);
        }

        QK_mma.pointwise([&](float el, int index) {
            global_row = tile_y * mma_warp_tile::N_tile + (index / mma_warp_tile::M_thread) * mma_warp_tile::N_warp + QK_mma.thread_y;
            global_col = tile_x * mma_warp_tile::M_tile + (index % mma_warp_tile::M_thread) * mma_warp_tile::M_warp + QK_mma.thread_x;

            if (global_row >= N || global_col >= M)
                return 0.f;

            bias = has_attn_bias ? attn_bias_[global_row][global_col] : 0.f;

            if (causal && (global_row < (global_col - MN_DIFF)))
                return 0.f;

            return __expf((scale * el + bias) - scale); 
        });

        QK_mma.store_transpose(A_sm.smem);

        __syncthreads();

        // Second matmul:
        V_sm.load(V_, 0, tile_x, M);

        __syncthreads();

        for (int d = 0; d < mma_warp_tile::M_tile; d++) {
            out_mma.mma(A_sm.smem, V_sm.smem, d);
        }

        __syncthreads();
    }

    out_mma.store(O_sm.smem);

    __syncthreads();

    out_mma.store_rowsum(L_, tile_y, N);

    O_sm.store(O_, 0, tile_y, N);
}

// forwards c++ function

std::vector<at::Tensor> flash_cosine_sim_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    float scale,
    bool causal
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int N = Q.size(2);
    const int M = K.size(2);
    const int D = Q.size(3);
    const int E = V.size(3);

    auto options = torch::TensorOptions().device(device_of(Q)).dtype(torch::kFloat);

    auto O = at::empty({batch, heads, N, E}, options);
    auto L = at::empty({batch, heads, N}, options);

    const dim3 threads_per_block(256);
    const dim3 blocks(cdiv(N, mma_warp_tile::N_tile), batch * heads);

    const int max_feature_dimension = max(D, E);

    const unsigned shared_mem_size = (mma_warp_tile::N_tile * max_feature_dimension +
                                      mma_warp_tile::M_tile * max_feature_dimension +
                                      mma_warp_tile::N_tile * mma_warp_tile::M_tile) * sizeof(float);

    const bool has_attn_bias = !!attn_bias.numel();
    const bool has_mask = !!mask.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        forward_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            ACCESSOR(Q, 4, float),
            ACCESSOR(K, 4, float),
            ACCESSOR(V, 4, float),
            ACCESSOR(O, 4, float),
            ACCESSOR(L, 3, float),
            ACCESSOR(mask, 2, bool),
            ACCESSOR(attn_bias, 3, float),
            scale,
            causal,
            has_mask,
            has_attn_bias
        );
    }));

    // handle error
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    return { O, L };
}

// backward kernel

// backwards preprocess

// calculate do_scaled = rowsum(do * o)
// done by @ptillet at https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

template <typename scalar_t>
__global__ void backward_calculate_do_scaled(
    const PackedAccessor<scalar_t, 4> d_out,
    const PackedAccessor<scalar_t, 4> o,
          PackedAccessor<scalar_t, 3> do_scaled
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

    float val = 0.0f;

    extern __shared__ float _shared_mem_preprocess[];

    float* sm_do_scaled = (float*) &_shared_mem_preprocess;

    auto do_ = d_out[batch_idx][head_idx][seq_idx];
    auto o_ = o[batch_idx][head_idx][seq_idx];
    auto do_scaled_ = do_scaled[batch_idx][head_idx];

    // load into shared memory

    if (dim_idx < v_dim)
        val = do_[dim_idx] * o_[dim_idx];

    // warp shuffle reduce

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane_id == 0)
        sm_do_scaled[warp_id] = val;

    __syncthreads();

    if (warp_id == 0) {
        if (dim_idx < (blockDim.x / 32)) {
            val = sm_do_scaled[lane_id];
        } else{
            val = 0;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (dim_idx == 0) {
            do_scaled_[seq_idx] = val;
        }
    }
}

// main backward kernel

template <typename scalar_t>
__global__ void backward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<scalar_t, 3> attn_bias,
          PackedAccessor<scalar_t, 4> dq,
          PackedAccessor<scalar_t, 4> dk,
          PackedAccessor<scalar_t, 4> dv,
          PackedAccessor<scalar_t, 3> d_attn_bias,
    const PackedAccessor<scalar_t, 4> d_out,
    const PackedAccessor<scalar_t, 3> do_scaled,
    const PackedAccessor<scalar_t, 3> l,
    const float scale,
    const bool causal,
    const bool has_mask,
    const bool has_attn_bias,
    const int row_tile_size,
    const int col_tile_size,
    const int row_tiles,
    const int col_tiles
) {

    const int batch = q.size(0);
    const int head = q.size(1);

    const int batch_idx = blockIdx.x / head;
    const int head_idx = blockIdx.x % head;

    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const int num_col_tiles = cdiv(k_seq_len, col_tile_size);
    const int num_row_tiles = cdiv(q_seq_len, row_tile_size);

    const int row_tiles_idx = blockIdx.y / col_tiles;
    const int col_tiles_idx = blockIdx.y % col_tiles;

    const int col_tile_idx = threadIdx.x;
    const int row_tile_idx = threadIdx.y;

    const int lane_id = threadIdx.x & 31;

    const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int tpb = blockDim.x * blockDim.y;

    const int k_total_el = k_dim * col_tile_size;
    const int v_total_el = v_dim * col_tile_size;

    const int sm_q_offset = row_tile_idx * k_dim;
    const int sm_k_offset = col_tile_idx * k_dim;
    const int sm_v_offset = col_tile_idx * v_dim;
    const int sm_o_offset = row_tile_idx * v_dim;

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][head_idx];
    auto v_ = v[batch_idx][head_idx];
    auto dq_ = dq[batch_idx][head_idx];
    auto dk_ = dk[batch_idx][head_idx];
    auto dv_ = dv[batch_idx][head_idx];
    auto ds_ = d_attn_bias[head_idx];
    auto do_scaled_ = do_scaled[batch_idx][head_idx];
    auto l_ = l[batch_idx][head_idx];
    auto do_ = d_out[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // handle attention bias

    auto attn_bias_ = has_attn_bias ? attn_bias[head_idx] : attn_bias[0];

    // some variables

    int col_tiles_offset, row_tiles_offset;
    int global_col, global_row;
    bool should_calculate_attn, should_calculate_row, should_calculate_col;

    // shared memory

    extern __shared__ float _shared_mem[];

    float* sm_q = (float*) &_shared_mem;
    float* sm_k = (float*) &sm_q[row_tile_size * k_dim];
    float* sm_v = (float*) &sm_k[col_tile_size * k_dim];
    float* sm_l = (float*) &sm_v[col_tile_size * v_dim];
    float* sm_do_scaled = (float*) &sm_l[row_tile_size];
    float* sm_do = (float*) &sm_do_scaled[row_tile_size];

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * col_tile_size;
        global_col = col_tiles_offset + col_tiles_idx * col_tile_size + col_tile_idx;
        should_calculate_col = global_col < k_seq_len && (!has_mask || mask_[global_col]);

        // coalesced reads
        // cleanup later

        for (
            int offset = 0;
            offset < k_total_el;
            offset += tpb
        ) {
            int sm_idx = offset + thread_idx;
            int gmem_seq_idx = sm_idx / k_dim;
            int gmem_dim_idx = sm_idx % k_dim;

            if (offset < k_total_el)
                sm_k[sm_idx] = k_[col_tiles_offset + col_tiles_idx * col_tile_size + gmem_seq_idx][gmem_dim_idx];
        }

        for (
            int offset = 0;
            offset < v_total_el;
            offset += tpb
        ) {
            int sm_idx = offset + thread_idx;
            int gmem_seq_idx = sm_idx / v_dim;
            int gmem_dim_idx = sm_idx % v_dim;

            if (offset < v_total_el)
                sm_v[sm_idx] = v_[col_tiles_offset + col_tiles_idx * col_tile_size + gmem_seq_idx][gmem_dim_idx];
        }

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * row_tile_size;
            global_row = row_tiles_offset + row_tiles_idx * row_tile_size + row_tile_idx;
            should_calculate_row = global_row < q_seq_len;

            should_calculate_attn = should_calculate_row &&
                                    should_calculate_col &&
                                    ( !causal ||
                                      (causal && (global_row >= (global_col - k_seq_len + q_seq_len))));

            for (
                int d = col_tile_idx;
                d < k_dim;
                d += col_tile_size
            ) {
                sm_q[row_tile_idx * k_dim + d] = q_[row_tiles_offset + row_tiles_idx * row_tile_size + row_tile_idx][d];
            }

            for (
                int d = col_tile_idx;
                d < v_dim;
                d += col_tile_size
            ) {
                sm_do[row_tile_idx * v_dim + d] = do_[row_tiles_offset + row_tiles_idx * row_tile_size + row_tile_idx][d];
            }

            if (col_tile_idx == 0) {
                sm_do_scaled[row_tile_idx] = do_scaled_[global_row];
                sm_l[row_tile_idx] = l_[global_row];
            }

            __syncthreads();

            float attn = 0;
            float row_sum = 0;
            float dp = 0;

            if (should_calculate_attn) {
                for (int d = 0; d < k_dim; d++) {
                    // dmod is a "hacky" way to avoid bank register conflicts from @ahennequ
                    int dmod = (d + lane_id) % k_dim;
                    attn += sm_q[sm_q_offset + dmod] * sm_k[sm_k_offset + dmod];
                }

                attn *= scale;

                if (has_attn_bias) {
                    attn += attn_bias_[global_row][global_col];
                }

                attn -= scale;
                attn = __expf(attn);

                row_sum = sm_l[row_tile_idx];

                if (row_sum > 1e-8)
                    attn /= row_sum;

                for (int d = 0; d < v_dim; d++) {
                    // accumulate dv to global mem

                    atomicAdd((float*) &dv_[global_col][d], sm_do[sm_o_offset + d] * attn);

                    // calculate dp

                    dp += sm_do[sm_o_offset + d] * sm_v[sm_v_offset + d];
                }
            }

            // calculate dS

            float dS = 0;

            if (should_calculate_attn) {
                float D = sm_do_scaled[row_tile_idx];

                dS = attn * (dp - D);

                if (has_attn_bias) {
                    atomicAdd((float*) &ds_[global_row][global_col], dS);
                }
            }

            __syncthreads();

            // accumulate dq and dk to global mem

            if (should_calculate_attn) {
                dS *= scale;

                for (int d = 0; d < k_dim; d++) {
                    atomicAdd((float*) &dq_[global_row][d], dS * sm_k[sm_k_offset + d]);

                    atomicAdd((float*) &dk_[global_col][d], dS * sm_q[sm_q_offset + d]);
                }
            }

            __syncthreads();
        }
    }
}

// backwards c++ function

std::vector<torch::Tensor> flash_cosine_sim_attention_backward(
    torch::Tensor d_out,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor d_attn_bias,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    float scale,
    bool causal,
    int row_tile_size,
    int col_tile_size,
    int row_tiles,
    int col_tiles
) {
    auto query_device = device_of(q);

    const at::cuda::OptionalCUDAGuard device_guard(query_device);

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seq   = q.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const bool has_attn_bias = !!d_attn_bias.numel();
    const bool has_mask = !!mask.numel();

    auto options = torch::TensorOptions().device(query_device).dtype(torch::kFloat);

    // setup dq, dk, dv

    auto do_scaled = at::empty_like(l, options);

    auto dq = at::zeros_like(q, options);
    auto dk = at::zeros_like(k, options);
    auto dv = at::zeros_like(v, options);

    // setup backwards preprocess call

    const dim3 backwards_preprocess_threads_per_block(next_multiple_of(v_dim, 32));

    const dim3 backwards_preprocess_blocks(batch * heads, seq);

    const unsigned backwards_preprocess_shared_mem_size = cdiv(v_dim, 32) * sizeof(float);

    // setup backwards call

    const dim3 backwards_threads_per_block(col_tile_size, row_tile_size);
    const dim3 backwards_blocks(batch * heads, row_tiles * col_tiles);

    const unsigned backwards_shared_mem_size = (  (row_tile_size + col_tile_size) * k_dim +      // q, k
                                                  (row_tile_size + col_tile_size) * v_dim +      // v, do
                                                  (row_tile_size + col_tile_size)                // l, do_scaled
                                                ) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        backward_calculate_do_scaled<scalar_t><<<backwards_preprocess_blocks, backwards_preprocess_threads_per_block, backwards_preprocess_shared_mem_size>>>(
            ACCESSOR(d_out, 4, scalar_t),
            ACCESSOR(o, 4, scalar_t),
            ACCESSOR(do_scaled, 3, scalar_t)
        );

        backward_kernel<scalar_t><<<backwards_blocks, backwards_threads_per_block, backwards_shared_mem_size>>>(
            ACCESSOR(q, 4, scalar_t),
            ACCESSOR(k, 4, scalar_t),
            ACCESSOR(v, 4, scalar_t),
            ACCESSOR(mask, 2, bool),
            ACCESSOR(attn_bias, 3, scalar_t),
            ACCESSOR(dq, 4, scalar_t),
            ACCESSOR(dk, 4, scalar_t),
            ACCESSOR(dv, 4, scalar_t),
            ACCESSOR(d_attn_bias, 3, scalar_t),
            ACCESSOR(d_out, 4, scalar_t),
            ACCESSOR(do_scaled, 3, scalar_t),
            ACCESSOR(l, 3, scalar_t),
            scale,
            causal,
            has_mask,
            has_attn_bias,
            row_tile_size,
            col_tile_size,
            row_tiles,
            col_tiles
        );
    }));

    cudaDeviceSynchronize();

    // handle error

    CHECK_LAST_CUDA_ERROR();

    return {dq, dk, dv};
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
