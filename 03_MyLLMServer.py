import torch
from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Initialize Flask app
app = Flask(__name__)

# Login to Huggingface
hf_access_key = "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"  # noqa
login(hf_access_key)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config
)


# Function to compute embedding
def compute_embedding(text):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden_states = hidden_states * attention_mask
    sum_embeddings = masked_hidden_states.sum(dim=1)
    non_pad_tokens = attention_mask.sum(dim=1)
    embedding = sum_embeddings / non_pad_tokens.clamp(min=1e-9)
    return embedding[0].cpu().numpy()


# Function to generate only new tokens as text
def generate_text(text, temperature=0.1, max_new_tokens=25):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].cuda(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            attention_mask=inputs['attention_mask'].cuda(),
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens (excluding input text tokens)
    input_length = inputs['input_ids'].shape[1]
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text


# Flask route for computing embedding or generating text
@app.route("/my_llama3", methods=["POST"])
def embedding_endpoint():
    data = request.json
    text = data.get("text", "")
    returns_embedding = data.get("returns_embedding", True)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if returns_embedding:
        # Compute and return embedding
        embedding = compute_embedding(text)
        return jsonify({"embedding": embedding.tolist()})
    else:
        # Generate and return text
        temperature = data.get("temperature", 0.1)
        max_new_tokens = data.get("max_new_tokens", 50)
        generated_text = generate_text(text, temperature=temperature, max_new_tokens=max_new_tokens)
        return jsonify({"generated_text": generated_text})


# To test the server for embedding computation:
"""
curl -X POST http://localhost:7777/my_llama3 \
     -H "Content-Type: application/json" \
     -d '{"text": "Sample text to test embedding or generation", "returns_embedding": true}'
"""

# To test the server for text generation:
"""
curl -X POST http://localhost:7777/my_llama3 \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Sample text to generate inference output",
           "returns_embedding": false,
           "temperature": 0.7,
           "max_new_tokens": 50
         }'
"""

# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777)
