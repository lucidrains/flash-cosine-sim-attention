from flash_cosine_sim_attention.flash_cosine_sim_attention import flash_cosine_sim_attention, plain_cosine_sim_attention, l2norm_tensors, debug

try:
    from flash_cosine_sim_attention.triton_flash_cosine_sim_attention import triton_flash_cosine_sim_attention
except ImportError:
    triton_flash_cosine_sim_attention = None
