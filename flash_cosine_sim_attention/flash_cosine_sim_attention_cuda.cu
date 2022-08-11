#include <torch/extension.h>

torch::Tensor flash_cosine_sim_attention_forward(torch::Tensor q, torch::Tensor k,  torch::Tensor v) {
    auto o = torch::zeros_like(q);
    return o;
}

std::vector<torch::Tensor> flash_cosine_sim_attention_backward(torch::Tensor grad_o, torch::Tensor o, torch::Tensor l, torch::Tensor q,  torch::Tensor k,  torch::Tensor v) {
    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_cosine_sim_attention_forward, "Flash Cosine-Sim Attention Forward");
    m.def("backward", &flash_cosine_sim_attention_backward, "Flash Cosine-Sim Attention Backward");
}
