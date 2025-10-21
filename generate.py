# src/generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "outputs/final/peft_bandeira"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

prompt = "Vou-me embora pra Pasárgada\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
