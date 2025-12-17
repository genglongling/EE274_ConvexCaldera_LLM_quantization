import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./llama2_7b_scl_scalar8"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map={"": "cuda"},
)

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model(**inputs)
    logits = out.logits

print("logits has NaN?", torch.isnan(logits).any().item())
print("logits has Inf?", torch.isinf(logits).any().item())
