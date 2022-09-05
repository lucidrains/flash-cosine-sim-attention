#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>

// constants

__constant__ int warp_size = 32;

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

__device__ int cdiv(int numer, int denom) {
    return (numer + denom - 1) / denom;
}

__device__ int next_pow_2(int n) {
    int i = 1;
    while(i < n)
        i *= 2;
    return i;
}

__device__ void warp_reduce(volatile float* sm, int tid, int max) {
    for (int s = warp_size; s > 0; s>>=1) {
        if ((tid + s) >= max)
            continue;

        sm[tid] += sm[tid + s];
    }
}

bool divisible_by(int num, int denom) {
    return (num % denom) == 0;
}

// forward kernel

template <typename scalar_t>
__global__ void forward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<bool, 2> mask,
    const PackedAccessor<scalar_t, 3> attn_bias,
          PackedAccessor<scalar_t, 4> o,
          PackedAccessor<scalar_t, 3> l,
    const float scale,
    const bool causal,
    const bool has_attn_bias,
    const int q_block_size,
    const int k_block_size
) {
    const int batch = q.size(0);
    const int head = q.size(1);

    const int batch_idx = blockIdx.x / head;
    const int head_idx = blockIdx.x % head;

    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const int num_col_tiles = cdiv(k_seq_len, k_block_size);
    const int num_row_tiles = cdiv(q_seq_len, q_block_size);

    const int tile_size = blockDim.x;
    const int q_tiles = q_block_size / tile_size;
    const int k_tiles = k_block_size / tile_size;

    const int row_tiles_idx = blockIdx.y / k_tiles;
    const int col_tiles_idx = blockIdx.y % k_tiles;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

    const int sm_q_offset = row_tile_idx * k_dim;
    const int sm_k_offset = col_tile_idx * k_dim;
    const int sm_v_offset = col_tile_idx * v_dim;
    const int sm_o_offset = row_tile_idx * v_dim;

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][head_idx];
    auto v_ = v[batch_idx][head_idx];
    auto o_ = o[batch_idx][head_idx];
    auto l_ = l[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // handle attention bias

    auto attn_bias_ = has_attn_bias ? attn_bias[head_idx] : attn_bias[0];

    // shared memory

    extern __shared__ float _shared_mem[];

    float* sm_q = (float*) &_shared_mem;
    float* sm_k = (float*) &sm_q[tile_size * k_dim];
    float* sm_v = (float*) &sm_k[tile_size * k_dim];

    // some variable

    int col_tiles_offset, row_tiles_offset;
    int global_col, global_row;
    bool should_calculate_attn, should_calculate_row, should_calculate_col;

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;
        global_col = col_tiles_offset + col_tiles_idx * tile_size + col_tile_idx;
        should_calculate_col = global_col < k_seq_len && mask_[global_col];

        if (should_calculate_col) {
            for (
                int d = row_tile_idx;
                d < k_dim;
                d += tile_size
            ) {
                sm_k[sm_k_offset + d] = k_[global_col][d];
            }

            for (
                int d = row_tile_idx;
                d < v_dim;
                d += tile_size
            ) {
                sm_v[sm_v_offset + d] = v_[global_col][d];
            }
        }

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
            global_row = row_tiles_offset + row_tiles_idx * tile_size + row_tile_idx;
            should_calculate_row = global_row < q_seq_len;

            should_calculate_attn = should_calculate_row &&
                                    should_calculate_col &&
                                    ( !causal ||
                                      (causal && (global_row >= (global_col - k_seq_len + q_seq_len))));

            if (should_calculate_row) {
                for (
                    int d = col_tile_idx;
                    d < k_dim;
                    d += tile_size
                ) {
                    sm_q[sm_q_offset + d] = q_[global_row][d];
                }
            }

            __syncthreads();

            if (should_calculate_attn) {
                float attn = 0;
                for (int d = 0; d < k_dim; d++) {
                    // dmod is a "hacky" way to avoid bank register conflicts from @ahennequ
                    int dmod = (d + (threadIdx.x % warp_size)) % k_dim;
                    attn += sm_q[sm_q_offset + dmod] * sm_k[sm_k_offset + dmod];
                }

                attn *= scale;

                if (has_attn_bias) {
                    attn += attn_bias_[global_row][global_col];
                }

                attn -= scale;
                attn = __expf(attn);

                atomicAdd((float*) &l_[global_row], attn);

                float exp_weighted_value;

                for (int d = 0; d < v_dim; d++) {
                    exp_weighted_value = attn * sm_v[sm_v_offset + d];
                    atomicAdd((float*) &o_[global_row][d], exp_weighted_value);
                }
            }

            __syncthreads();
        }
    }
}

// forwards c++ function

void flash_cosine_sim_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    float scale,
    bool causal,
    int q_block_size,
    int k_block_size,
    int tile_size
) {
    assert(("tile size needs to be 32 or less", tile_size <= 32));
    assert(("query block size needs to be divisible by tile size", divisible_by(q_block_size, tile_size)));
    assert(("key block size needs to be divisible by tile size", divisible_by(k_block_size, tile_size)));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(o));

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);
    const bool has_attn_bias = !!attn_bias.numel();

    const dim3 threads_per_block(tile_size, tile_size);
    const dim3 blocks(batch * heads, (q_block_size / tile_size) * (k_block_size / tile_size));

    const unsigned shared_mem_size = tile_size * (2 * k_dim + v_dim) * sizeof(float); // q, k, v

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            ACCESSOR(q, 4, scalar_t),
            ACCESSOR(k, 4, scalar_t),
            ACCESSOR(v, 4, scalar_t),
            ACCESSOR(mask, 2, bool),
            ACCESSOR(attn_bias, 3, scalar_t),
            ACCESSOR(o, 4, scalar_t),
            ACCESSOR(l, 3, scalar_t),
            scale,
            causal,
            has_attn_bias,
            q_block_size,
            k_block_size
        );
    }));

    cudaDeviceSynchronize();

    // handle error

    CHECK_LAST_CUDA_ERROR();

    return;
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

    extern __shared__ float _shared_mem_preprocess[];

    float* sm_do_scaled = (float*) &_shared_mem_preprocess;

    auto do_ = d_out[batch_idx][head_idx][seq_idx];
    auto o_ = o[batch_idx][head_idx][seq_idx];
    auto do_scaled_ = do_scaled[batch_idx][head_idx];

    // load into shared memory

    sm_do_scaled[dim_idx] = do_[dim_idx] * o_[dim_idx];

    __syncthreads();

    // better sum reduce

    for (int s = next_pow_2(v_dim) / 2; s > warp_size; s>>=1) {

        if ((dim_idx + s) < v_dim)
            sm_do_scaled[dim_idx] += sm_do_scaled[dim_idx + s];

        __syncthreads();
    }

    warp_reduce(sm_do_scaled, dim_idx, v_dim);

    __syncthreads();

    if (dim_idx == 0) {
        do_scaled_[seq_idx] = sm_do_scaled[0];
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
    const bool has_attn_bias,
    const int q_block_size,
    const int k_block_size
) {

    const int batch = q.size(0);
    const int head = q.size(1);

    const int batch_idx = blockIdx.x / head;
    const int head_idx = blockIdx.x % head;

    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const int num_col_tiles = cdiv(k_seq_len, k_block_size);
    const int num_row_tiles = cdiv(q_seq_len, q_block_size);

    const int tile_size = blockDim.x;
    const int q_tiles = q_block_size / tile_size;
    const int k_tiles = k_block_size / tile_size;

    const int row_tiles_idx = blockIdx.y / k_tiles;
    const int col_tiles_idx = blockIdx.y % k_tiles;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

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
    float* sm_k = (float*) &sm_q[tile_size * k_dim];
    float* sm_v = (float*) &sm_k[tile_size * k_dim];
    float* sm_l = (float*) &sm_v[tile_size * v_dim];
    float* sm_do_scaled = (float*) &sm_l[tile_size];
    float* sm_do = (float*) &sm_do_scaled[tile_size];

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;
        global_col = col_tiles_offset + col_tiles_idx * tile_size + col_tile_idx;
        should_calculate_col = global_col < k_seq_len && mask_[global_col];

        for (
            int d = row_tile_idx;
            d < k_dim;
            d += tile_size
        ) {
            sm_k[sm_k_offset + d] = k_[global_col][d];
        }

        for (
            int d = row_tile_idx;
            d < v_dim;
            d += tile_size
        ) {
            sm_v[sm_v_offset + d] = v_[global_col][d];
        }

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
            global_row = row_tiles_offset + row_tiles_idx * tile_size + row_tile_idx;
            should_calculate_row = global_row < q_seq_len;

            should_calculate_attn = should_calculate_row &&
                                    should_calculate_col &&
                                    ( !causal ||
                                      (causal && (global_row >= (global_col - k_seq_len + q_seq_len))));

            for (
                int d = col_tile_idx;
                d < k_dim;
                d += tile_size
            ) {
                sm_q[sm_q_offset + d] = q_[global_row][d];
            }

            for (
                int d = col_tile_idx;
                d < v_dim;
                d += tile_size
            ) {
                sm_do[sm_o_offset + d] = do_[global_row][d];
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
                    int dmod = (d + (threadIdx.x % warp_size)) % k_dim;
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

void flash_cosine_sim_attention_backward(
    torch::Tensor d_out,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor dq,
    torch::Tensor dk,
    torch::Tensor dv,
    torch::Tensor d_attn_bias,
    torch::Tensor do_scaled,
    torch::Tensor mask,
    torch::Tensor attn_bias,
    float scale,
    bool causal,
    int q_block_size,
    int k_block_size,
    int tile_size
) {
    assert(("tile size needs to be 32 or less", tile_size <= 32));
    assert(("query block size needs to be divisible by tile size", divisible_by(q_block_size, tile_size)));
    assert(("key block size needs to be divisible by tile size", divisible_by(k_block_size, tile_size)));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dq));

    const int batch = dq.size(0);
    const int heads = dq.size(1);

    const int seq   = dq.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);
    const bool has_attn_bias = !!attn_bias.numel();

    // setup backwards preprocess call

    const dim3 backwards_preprocess_threads_per_block(v_dim);
    const dim3 backwards_preprocess_blocks(batch * heads, seq);

    const unsigned backwards_preprocess_shared_mem_size = v_dim * sizeof(float);

    const dim3 backwards_threads_per_block(tile_size, tile_size);
    const dim3 backwards_blocks(batch * heads, (q_block_size / tile_size) * (k_block_size / tile_size));

    const unsigned backwards_shared_mem_size = (  tile_size * 2 * k_dim +      // q, k
                                                  tile_size * v_dim * 2 +      // v, do
                                                  tile_size * 2                // l, do_scaled
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
            has_attn_bias,
            q_block_size,
            k_block_size
        );
    }));

    cudaDeviceSynchronize();

    // handle error

    CHECK_LAST_CUDA_ERROR();

    return;
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
