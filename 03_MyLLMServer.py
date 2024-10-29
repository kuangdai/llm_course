import pickle

import faiss
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Initialize Flask app
app = Flask(__name__)

#########
# Model #
#########

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

###############
# Poetry data #
###############

# Load the .npz file with allow_pickle=True
loaded_npz = np.load('data/input/poetry_data_clean.npz', allow_pickle=True)

# Reconstruct the DataFrame using the saved data and columns
df = pd.DataFrame(loaded_npz['df'], columns=loaded_npz['columns'])

# Load Faiss index
faiss_index = faiss.read_index("data/knowledge_bases/poetry_faiss.index")

# Load keyword data
with open('data/knowledge_bases/poetry_unique_keywords.pkl', 'rb') as f:
    unique_keywords_list = pickle.load(f)
with open('data/knowledge_bases/poetry_forward_mapping.pkl', 'rb') as f:
    forward_mapping = pickle.load(f)
with open('data/knowledge_bases/poetry_inverse_mapping.pkl', 'rb') as f:
    inverse_mapping = pickle.load(f)

# Load NetworkX graph
with open("data/knowledge_bases/poetry_keyword_graph.gpickle", "rb") as f:
    kw_graph = pickle.load(f)


def format_poems(idx, include_keywords=True, include_tags=False):
    """ format poems """
    if hasattr(idx, "__len__"):
        return [format_poems(i, include_keywords, include_tags) for i in idx]
    it = df.iloc[idx]
    res = f'{it["Title"]}\n{it["Poet"]}\n\n{it["Poem"]}'
    if include_keywords:
        res += f'\n\nKeywords: {unique_keywords_list[forward_mapping[idx]]}'
    if it["Tags"] and include_tags:
        res += f'\n\nNotes: {it["Tags"]}'
    return res


###################
# Text Generation #
###################

def generate_kernel(text, temperature=0.1, max_new_tokens=25):
    """Function to generate only new tokens as text"""
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
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


@app.route("/generate", methods=["POST"])
def generate():
    """Route to generate"""
    data = request.json
    text = data.get("text", "")
    temperature = data.get("temperature", 0.1)
    max_new_tokens = data.get("max_new_tokens", 50)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate and return text
    generated_text = generate_kernel(text, temperature=temperature, max_new_tokens=max_new_tokens)
    return jsonify({"generated_text": generated_text})


# To test the server for text generation:
"""
curl -X POST http://localhost:7777/generate \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello. How are you today?",
           "temperature": 0.7,
           "max_new_tokens": 50
         }'
"""


#############################
# Similarity-based Retrival #
#############################

def retrieve_faiss_kernel(text, k, flatten=True):
    """Function to retrieve from faiss index"""
    # Compute embedding
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden_states = hidden_states * attention_mask
    sum_embeddings = masked_hidden_states.sum(dim=1)
    non_pad_tokens = attention_mask.sum(dim=1)
    embedding = sum_embeddings / non_pad_tokens.clamp(min=1e-9)

    # Retrieve indices
    distances, indices = faiss_index.search(embedding.cpu().numpy(), k=k)

    # Convert to text
    poems = format_poems(indices, include_keywords=True)
    if flatten:
        poems = "\n\n----------------\n\n".join(poems)
    return poems


@app.route("/retrieve_faiss", methods=["POST"])
def retrieve_faiss():
    """Route to retrieve from faiss index"""
    data = request.json
    text = data.get("text", "")
    k = data.get("k", 1)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Compute and return embedding
    return jsonify({"retrieved_poems": retrieve_faiss_kernel(text, k, flatten=True)})


# To test the server for similarity retrival:
"""
curl -X POST http://localhost:7777/retrieve_faiss \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"I don't know how you were diverted\\nYou were perverted too\", \"k\": 1}"
"""

##########################
# Keyword-based Retrival #
##########################


# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777)
