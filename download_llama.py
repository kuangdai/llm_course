import json

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Read HF_ACCESS_KEY into hf_access_key
with open("api_keys.json", "r") as file:
    hf_access_key = json.load(file).get("HF_ACCESS_KEY")

# Login to HuggingFace
login(hf_access_key)

# Create a BitsAndBytesConfig for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Change this to `False` to disable quantization
    bnb_4bit_use_double_quant=True,  # Optional for performance
    bnb_4bit_quant_type='nf4',  # Normal floating-point 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16  # Set compute dtype to float16 for faster inference
)

# Model name--you can change to many huggingface models
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config
)
