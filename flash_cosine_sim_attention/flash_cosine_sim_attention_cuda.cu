#include <cuda.h>
#include <cuda_runtime.h>
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
        if ((tid + s) < max)
            sm[tid] += sm[tid + s];
    }
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
    const int q_block_size,
    const int k_block_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const int num_col_tiles = cdiv(k_seq_len, k_block_size);
    const int num_row_tiles = cdiv(q_seq_len, q_block_size);

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

    const bool has_attn_bias = attn_bias.size(1) > 0 && attn_bias.size(2) > 0;
    auto attn_bias_ = has_attn_bias ? attn_bias[head_idx] : attn_bias[0];

    // shared memory

    extern __shared__ float _shared_mem[];

    float* sm_q = (float*) &_shared_mem;
    float* sm_k = (float*) &sm_q[q_block_size * k_dim];
    float* sm_v = (float*) &sm_k[k_block_size * k_dim];
    float* sm_l = (float*) &sm_v[k_block_size * v_dim];
    float* sm_o = (float*) &sm_l[q_block_size];

    // some variable

    int col_tiles_offset, row_tiles_offset;
    int global_col, global_row;
    bool should_calculate_attn, should_calculate_row, should_calculate_col;

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;
        global_col = col_tiles_offset + col_tile_idx;
        should_calculate_col = global_col < k_seq_len && mask_[global_col];

        if (should_calculate_col) {
            for (
                int d = row_tile_idx;
                d < k_dim;
                d += q_block_size
            ) {
                sm_k[sm_k_offset + d] = k_[global_col][d];
            }

            for (
                int d = row_tile_idx;
                d < v_dim;
                d += q_block_size
            ) {
                sm_v[sm_v_offset + d] = v_[global_col][d];
            }
        }

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
            global_row = row_tiles_offset + row_tile_idx;
            should_calculate_row = global_row < q_seq_len;

            should_calculate_attn = should_calculate_row &&
                                    should_calculate_col &&
                                    ( !causal ||
                                      (causal && (global_row >= (global_col - k_seq_len + q_seq_len))));

            if (should_calculate_row) {
                for (
                    int d = col_tile_idx;
                    d < k_dim;
                    d += k_block_size
                ) {
                    sm_q[sm_q_offset + d] = q_[global_row][d];
                }

                sm_l[row_tile_idx] = l_[global_row];

                for (
                    int d = col_tile_idx;
                    d < v_dim;
                    d += k_block_size
                ) {
                    sm_o[sm_o_offset + d] = o_[global_row][d];
                }
            }

            __syncthreads();

            if (should_calculate_attn) {
                float attn = 0;
                for (int d = 0; d < k_dim; d++) {
                    attn += sm_q[sm_q_offset + d] * sm_k[sm_k_offset + d];
                }

                attn *= scale;

                if (has_attn_bias) {
                    attn += attn_bias_[global_row][global_col];
                }

                attn -= scale;
                attn = __expf(attn);

                atomicAdd(&sm_l[row_tile_idx], attn);

                float exp_weighted_value;

                for (int d = 0; d < v_dim; d++) {
                    exp_weighted_value = attn * sm_v[sm_v_offset + d];
                    atomicAdd(&sm_o[sm_o_offset + d], exp_weighted_value);
                }
            }

            __syncthreads();

            if (should_calculate_row) {
                if (col_tile_idx == 0) {
                    l_[global_row] = sm_l[row_tile_idx];
                }

                for (
                    int d = col_tile_idx;
                    d < v_dim;
                    d += k_block_size
                ) {
                    o_[global_row][d] = sm_o[sm_o_offset + d];
                }
            }

            __syncthreads();
        }
    }
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
          PackedAccessor<scalar_t, 4> d_attn_bias,
    const PackedAccessor<scalar_t, 4> d_out,
    const PackedAccessor<scalar_t, 3> do_scaled,
    const PackedAccessor<scalar_t, 3> l,
    const float scale,
    const bool causal,
    const int q_block_size,
    const int k_block_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int q_seq_len = q.size(2);
    const int k_seq_len = k.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const int num_col_tiles = cdiv(k_seq_len, k_block_size);
    const int num_row_tiles = cdiv(q_seq_len, q_block_size);

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
    auto ds_ = d_attn_bias[batch_idx][head_idx];
    auto do_scaled_ = do_scaled[batch_idx][head_idx];
    auto l_ = l[batch_idx][head_idx];
    auto do_ = d_out[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // handle attention bias

    const bool has_attn_bias = attn_bias.size(1) > 0 && attn_bias.size(2) > 0;
    auto attn_bias_ = has_attn_bias ? attn_bias[head_idx] : attn_bias[0];

    // some variables

    int col_tiles_offset, row_tiles_offset;
    int global_col, global_row;
    bool should_calculate_attn, should_calculate_row, should_calculate_col;

    // shared memory

    extern __shared__ float _shared_mem[];

    float* sm_q = (float*) &_shared_mem;
    float* sm_k = (float*) &sm_q[q_block_size * k_dim];
    float* sm_v = (float*) &sm_k[k_block_size * k_dim];
    float* sm_l = (float*) &sm_v[k_block_size * v_dim];
    float* sm_do_scaled = (float*) &sm_l[q_block_size];
    float* sm_do = (float*) &sm_do_scaled[q_block_size];

    float* sm_dq = (float*) &sm_do[q_block_size * v_dim];
    float* sm_dk = (float*) &sm_dq[q_block_size * k_dim];
    float* sm_dv = (float*) &sm_dk[k_block_size * k_dim];

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;
        global_col = col_tiles_offset + col_tile_idx;
        should_calculate_col = global_col < k_seq_len && mask_[global_col];

        for (
            int d = row_tile_idx;
            d < k_dim;
            d += q_block_size
        ) {
            sm_k[sm_k_offset + d] = k_[global_col][d];
            sm_dk[sm_k_offset + d] = 0.;            
        }

        for (
            int d = row_tile_idx;
            d < v_dim;
            d += q_block_size
        ) {
            sm_v[sm_v_offset + d] = v_[global_col][d];
            sm_dv[sm_v_offset + d] = 0.;
        }

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
            global_row = row_tiles_offset + row_tile_idx;
            should_calculate_row = global_row < q_seq_len;

            should_calculate_attn = should_calculate_row &&
                                    should_calculate_col &&
                                    ( !causal ||
                                      (causal && (global_row >= (global_col - k_seq_len + q_seq_len))));

            for (
                int d = col_tile_idx;
                d < k_dim;
                d += k_block_size
            ) {
                sm_q[sm_q_offset + d] = q_[global_row][d];
                sm_dq[sm_q_offset + d] = dq_[global_row][d];
            }

            for (
                int d = col_tile_idx;
                d < v_dim;
                d += k_block_size
            ) {
                sm_do[sm_o_offset + d] = do_[global_row][d];
            }

            if (col_tile_idx == 0) {
                sm_do_scaled[row_tile_idx] = do_scaled_[global_row];
                sm_l[row_tile_idx] = l_[global_row];
            }

            __syncthreads();

            float attn = 0;
            float dp = 0;

            if (should_calculate_attn) {
                for (int d = 0; d < k_dim; d++) {
                    attn += sm_q[sm_q_offset + d] * sm_k[sm_k_offset + d];
                }

                attn *= scale;

                if (has_attn_bias) {
                    attn += attn_bias_[global_row][global_col];
                }

                attn -= scale;
                attn = __expf(attn);
                attn /= max(sm_l[row_tile_idx], 1e-10);

                for (int d = 0; d < v_dim; d++) {
                    // naively accumulate dv in shared mem

                    atomicAdd(&sm_dv[sm_v_offset + d], sm_do[sm_o_offset + d] * attn);

                    // calculate dp

                    dp += sm_do[sm_o_offset + d] * sm_v[sm_v_offset + d];
                }
            }

            // fetch precalculated D = rowsum(do * o)

            float D = sm_do_scaled[row_tile_idx];

            // calculate dS

            float dS = 0;

            if (should_calculate_attn) {
                dS = attn * (dp - D);

                if (has_attn_bias) {
                    ds_[global_row][global_col] = dS;
                }
            }

            __syncthreads();

            // calculate dq and dk and write to shared memoery

            if (should_calculate_attn) {
                dS *= scale;

                for (int d = 0; d < k_dim; d++) {
                    atomicAdd(&sm_dq[sm_q_offset + d], dS * sm_k[sm_k_offset + d]);

                    atomicAdd(&sm_dk[sm_k_offset + d], dS * sm_q[sm_q_offset + d]);
                }
            }

            __syncthreads();

            // write dq out to hbm

            if (col_tile_idx == 0 && should_calculate_row) {
                for (int d = 0; d < k_dim; d++) {
                    dq_[global_row][d] = sm_dq[sm_q_offset + d];
                }
            }

            __syncthreads();
        }

        // write dk and dv out to hbm

        if (should_calculate_col) {
            for (
                int d = row_tile_idx;
                d < k_dim;
                d += q_block_size
            ) {
                dk_[global_col][d] = sm_dk[sm_k_offset + d];
            }

            for (
                int d = row_tile_idx;
                d < v_dim;
                d += q_block_size
            ) {
                dv_[global_col][d] = sm_dv[sm_v_offset + d];
            }
        }

        __syncthreads();
    }
}

// main c++ function

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
    int k_block_size
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(o));

    const int batch = q.size(0);
    const int heads = q.size(1);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const dim3 threads_per_block(q_block_size, k_block_size);
    const dim3 blocks(batch, heads);
    const unsigned shared_mem_size = (( q_block_size + k_block_size) * k_dim +  // q, k
                                        k_block_size * v_dim +                  // v
                                        q_block_size * v_dim +                  // o
                                        q_block_size                            // l
                                      ) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            attn_bias.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale,
            causal,
            q_block_size,
            k_block_size
        );
    }));

    cudaDeviceSynchronize();

    // handle error

    CHECK_LAST_CUDA_ERROR();

    return;
}

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
    int k_block_size
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(dq));

    const int batch = dq.size(0);
    const int heads = dq.size(1);
    const int seq   = dq.size(2);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    // setup backwards preprocess call

    const dim3 backwards_preprocess_threads_per_block(v_dim);
    const dim3 backwards_preprocess_blocks(batch * heads, seq);

    const unsigned backwards_preprocess_shared_mem_size = v_dim * sizeof(float);

    const dim3 backwards_threads_per_block(q_block_size, k_block_size);
    const dim3 backwards_blocks(batch, heads);
    const unsigned backwards_shared_mem_size = (( q_block_size + k_block_size) * k_dim * 2 +   // q, k, dq, dk
                                                  k_block_size * v_dim * 2 +                   // v, dv
                                                  q_block_size * v_dim +                       // do
                                                  q_block_size                                 // l, do_scaled
                                                ) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        backward_calculate_do_scaled<scalar_t><<<backwards_preprocess_blocks, backwards_preprocess_threads_per_block, backwards_preprocess_shared_mem_size>>>(
            d_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            do_scaled.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );

        backward_kernel<scalar_t><<<backwards_blocks, backwards_threads_per_block, backwards_shared_mem_size>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            attn_bias.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dq.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dk.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            d_attn_bias.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            d_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            do_scaled.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale,
            causal,
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
