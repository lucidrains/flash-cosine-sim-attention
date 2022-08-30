#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// type alias

template <typename scalar_t, int dims>
using PackedAccessor = torch::PackedTensorAccessor32<scalar_t, dims, torch::RestrictPtrTraits>;

// forward kernel

template <typename scalar_t>
__global__ void forward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<bool, 2> mask,
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

    const int num_col_tiles = (k_seq_len + k_block_size - 1) / k_block_size;
    const int num_row_tiles = (q_seq_len + q_block_size - 1) / q_block_size;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][head_idx];
    auto v_ = v[batch_idx][head_idx];
    auto o_ = o[batch_idx][head_idx];
    auto l_ = l[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // shared memory

    extern __shared__ float _shared_mem[];
    float* shared_mem = (float*) _shared_mem;

    float* sm_q_block = (float*) &shared_mem[q_block_size * k_dim];
    float* sm_k_block = (float*) &sm_q_block[k_block_size * k_dim];
    float* sm_v_block = (float*) &sm_k_block[k_block_size * v_dim];
    float* sm_l_block = (float*) &sm_v_block[q_block_size];
    float* sm_o_block = (float*) &sm_v_block[q_block_size * v_dim];

    // some variable

    int col_tiles_offset, row_tiles_offset;
    bool is_last_col_tile;

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;

        if (col_tile_idx == 0) {
            for (int d = 0; d < k_dim; d++) {
                sm_k_block[col_tiles_offset + (col_tile_idx * k_dim) + d] = k_[col_tiles_offset + col_tile_idx][d];
            }

            for (int d = 0; d < v_dim; d++) {
                sm_v_block[col_tiles_offset + (col_tile_idx * v_dim) + d] = v_[col_tiles_offset + col_tile_idx][d];
            }
        }

        for (int j = 0; j < num_row_tiles; j++) {
            is_last_col_tile = (i == (num_col_tiles - 1));
            row_tiles_offset = j * q_block_size;

            if (row_tile_idx == 0) {
                for (int d = 0; d < k_dim; d++) {
                    sm_q_block[row_tiles_offset + (row_tile_idx * k_dim) + d] = q_[row_tiles_offset + row_tile_idx][d];
                }
            }

            float tmp = 0;
            for (int d = 0; d < k_dim; d++) {
                tmp += sm_q_block[(row_tile_idx * k_dim) + d] * sm_k_block[(col_tile_idx * k_dim) + d];
            }

            tmp = __expf(tmp);
            tmp *= scale;
            tmp -= scale;

            __syncthreads();
        }
    }
}

 // backward kernel

template <typename scalar_t>
__global__ void backward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
    const PackedAccessor<bool, 2> mask,
          PackedAccessor<scalar_t, 4> dq,
          PackedAccessor<scalar_t, 4> dk,
          PackedAccessor<scalar_t, 4> dv,
    const PackedAccessor<scalar_t, 4> grad_o,
    const PackedAccessor<scalar_t, 4> o,
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

    const int num_col_tiles = (k_seq_len + k_block_size - 1) / k_block_size;
    const int num_row_tiles = (q_seq_len + q_block_size - 1) / q_block_size;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

    auto q_ = q[batch_idx][head_idx];
    auto k_ = k[batch_idx][head_idx];
    auto v_ = v[batch_idx][head_idx];
    auto dq_ = dq[batch_idx][head_idx];
    auto dk_ = dk[batch_idx][head_idx];
    auto dv_ = dv[batch_idx][head_idx];
    auto o_ = o[batch_idx][head_idx];
    auto l_ = l[batch_idx][head_idx];
    auto grad_o_ = grad_o[batch_idx][head_idx];
    auto mask_ = mask[batch_idx];

    // some variables

    int col_tiles_offset, row_tiles_offset;
    bool is_last_col_tile;

    // shared memory

    extern __shared__ float _shared_mem[];
    float* shared_mem = (float*) _shared_mem;

    float* sm_q_block = (float*) &shared_mem[q_block_size * k_dim];
    float* sm_k_block = (float*) &sm_q_block[k_block_size * k_dim];
    float* sm_v_block = (float*) &sm_k_block[k_block_size * v_dim];
    float* sm_l_block = (float*) &sm_v_block[q_block_size];
    float* sm_o_block = (float*) &sm_l_block[q_block_size * v_dim];

    // loop

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;

        if (col_tile_idx == 0) {
            for (int d = 0; d < k_dim; d++) {
                sm_k_block[col_tiles_offset + (col_tile_idx * k_dim) + d] = k_[col_tiles_offset + col_tile_idx][d];
            }

            for (int d = 0; d < v_dim; d++) {
                sm_v_block[col_tiles_offset + (col_tile_idx * v_dim) + d] = v_[col_tiles_offset + col_tile_idx][d];
            }
        }

        for (int j = 0; j < num_row_tiles; j++) {
            is_last_col_tile = (i == (num_col_tiles - 1));
            row_tiles_offset = j * q_block_size;

            if (row_tile_idx == 0) {
                for (int d = 0; d < k_dim; d++) {
                    sm_q_block[row_tiles_offset + (row_tile_idx * k_dim) + d] = q_[row_tiles_offset + row_tile_idx][d];
                }
            }

            float tmp = 0;
            for (int d = 0; d < v_dim; d++) {
                tmp += q_[row_tiles_offset + row_tile_idx][d] * k_[col_tiles_offset + col_tile_idx][d];
            }

            tmp = __expf(tmp);
            tmp *= scale;
            tmp -= scale;

            __syncthreads();
        }
    }
}

// main c++ function

std::vector<torch::Tensor> flash_cosine_sim_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor mask,
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
    const unsigned shared_mem_size = ((q_block_size + k_block_size) * k_dim + k_block_size * v_dim + q_block_size + q_block_size * v_dim) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
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

    cudaError_t error = cudaGetLastError();

    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // output

    return {o, l};
}

std::vector<torch::Tensor> flash_cosine_sim_attention_backward(
    torch::Tensor grad_o,
    torch::Tensor o,
    torch::Tensor l,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask,
    float scale,
    bool causal,
    int q_block_size,
    int k_block_size
) {
    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dq));

    const int batch = dq.size(0);
    const int heads = dq.size(1);
    const int k_dim = k.size(3);
    const int v_dim = v.size(3);

    const dim3 threads_per_block(q_block_size, k_block_size);
    const dim3 blocks(batch, heads);
    const unsigned shared_mem_size = ((q_block_size + k_block_size) * k_dim + k_block_size * v_dim + q_block_size + q_block_size * v_dim) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        backward_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            dq.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dk.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
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

    cudaError_t error = cudaGetLastError();

    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // output

    return {dq, dk, dv};
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
