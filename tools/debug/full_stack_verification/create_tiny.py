import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

c = Qwen3Config(
    vocab_size=128,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    tie_word_embeddings=False,
)
c.architectures = ["Qwen3ForCausalLM"]
torch.manual_seed(23)
m = Qwen3ForCausalLM(c).to(torch.bfloat16)
m.save_pretrained("/tmp/tiny-qwen3")
print("saved random two-layer Qwen3; no downloads")
