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
          PackedAccessor<scalar_t, 4> o,
          PackedAccessor<scalar_t, 3> l,
    const float scale,
    const int q_block_size,
    const int k_block_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int q_seq_len = q.size(-2);
    const int k_seq_len = k.size(-2);
    const int dim = q.size(-1);

    const int num_col_tiles = (k_seq_len + k_block_size - 1) / k_block_size;
    const int num_row_tiles = (q_seq_len + q_block_size - 1) / q_block_size;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

    int col_tiles_offset, row_tiles_offset;

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
        }
    }
}

 // backward kernel

template <typename scalar_t>
__global__ void backward_kernel(
    const PackedAccessor<scalar_t, 4> q,
    const PackedAccessor<scalar_t, 4> k,
    const PackedAccessor<scalar_t, 4> v,
          PackedAccessor<scalar_t, 4> dq,
          PackedAccessor<scalar_t, 4> dk,
          PackedAccessor<scalar_t, 4> dv,
    const PackedAccessor<scalar_t, 4> grad_o,
    const PackedAccessor<scalar_t, 4> o,
    const PackedAccessor<scalar_t, 3> l,
    const float scale,
    const int q_block_size,
    const int k_block_size
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int q_seq_len = q.size(-2);
    const int k_seq_len = k.size(-2);
    const int dim = q.size(-1);

    const int num_col_tiles = (k_seq_len + k_block_size - 1) / k_block_size;
    const int num_row_tiles = (q_seq_len + q_block_size - 1) / q_block_size;

    const int row_tile_idx = threadIdx.x;
    const int col_tile_idx = threadIdx.y;

    int col_tiles_offset, row_tiles_offset;

    for (int i = 0; i < num_col_tiles; i++) {
        col_tiles_offset = i * k_block_size;

        for (int j = 0; j < num_row_tiles; j++) {
            row_tiles_offset = j * q_block_size;
        }
    }
}

// main c++ function

std::vector<torch::Tensor> flash_cosine_sim_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale,
    int q_block_size,
    int k_block_size
) {
    auto o = torch::zeros_like(q);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(o));

    auto l = torch::zeros_like(q).sum({-1,});

    const int batch = q.size(0);
    const int heads = q.size(1);

    const dim3 threads_per_block(q_block_size, k_block_size);
    const dim3 blocks(batch, heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale,
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
    float scale,
    int q_block_size,
    int k_block_size
) {
    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(dq));

    const int batch = dq.size(0);
    const int heads = dq.size(1);

    const dim3 threads_per_block(q_block_size, k_block_size);
    const dim3 blocks(batch, heads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        backward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dq.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dk.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale,
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
