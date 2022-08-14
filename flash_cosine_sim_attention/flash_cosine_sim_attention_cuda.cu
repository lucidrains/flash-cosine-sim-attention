#include <torch/extension.h>

// cuda kernels

template <typename scalar_t>
__global__ void forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> q,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> l,
    float scale) {

}

template <typename scalar_t>
__global__ void backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> q,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dq,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dk,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dv,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_o,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> l,
    float scale) {

}

// main c++ function

std::vector<torch::Tensor> flash_cosine_sim_attention_forward(torch::Tensor q, torch::Tensor k,  torch::Tensor v, float scale) {
    auto o = torch::zeros_like(q);
    auto l = torch::zeros_like(q).sum({-1,});

    const int blocks = 1;
    const int threads = 1;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_forward", ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale);
    }));

    return {o, l};
}

std::vector<torch::Tensor> flash_cosine_sim_attention_backward(torch::Tensor grad_o, torch::Tensor o, torch::Tensor l, torch::Tensor q,  torch::Tensor k,  torch::Tensor v, float scale) {
    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const int blocks = 1;
    const int threads = 1;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "forward_cosine_sim_attention_backward", ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dq.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dk.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dv.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            o.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            l.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            scale);
    }));

    return {dq, dk, dv};
}

// bind

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
