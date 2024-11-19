"""
This script sets up a local HTTP server using Flask to serve functionalities
developed in Jupyter notebooks (01_Inference.ipynb and 02_KnowledgeBase.ipynb),
focused on text generation and information retrieval.

Main features include:

1. **Text Generation**: Generates text based on a user-provided prompt using a
   pre-trained language model with 4-bit quantization for memory efficiency.

2. **Similarity-based Retrieval**: Retrieves similar poems from a FAISS index
   based on the embedding of the user query, allowing efficient nearest-neighbor
   search across a large dataset.

3. **Keyword-based Retrieval**: Uses a NetworkX bipartite graph to perform
   keyword-based poem retrieval, supporting multi-hop search and keyword inference
   to find relevant content through graph traversal.

The API provides endpoints for each of these functionalities, allowing integration
with other applications or user interfaces.
"""

import json
import os
import pickle
import shutil
import zipfile
from collections import defaultdict, deque

import faiss
import gdown
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import spacy

# Initialize Flask app
app = Flask(__name__)

###############
# Poetry Data #
###############

# Load poetry dataset
loaded_npz = np.load('data/input/poetry_data_clean.npz', allow_pickle=True)
df = pd.DataFrame(loaded_npz['df'], columns=loaded_npz['columns'])

# Define file paths
file_paths = [
    "data/knowledge_bases/poetry_faiss.index",
    "data/knowledge_bases/poetry_unique_keywords.pkl",
    "data/knowledge_bases/poetry_forward_mapping.pkl",
    "data/knowledge_bases/poetry_inverse_mapping.pkl",
    "data/knowledge_bases/poetry_keyword_graph.gpickle"
]

# Check if any file is missing
missing_files = [file for file in file_paths if not os.path.exists(file)]
if missing_files:
    print("Some files are missing. Downloading from Google Drive...")

    # Google Drive file ID and download URL
    file_id = "1hhi3Vc0ztcJIPdynE5XQ8L8kkJxnOYqE"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the zip file
    output_zip = "data/knowledge_bases_data.zip"
    gdown.download(gdrive_url, output_zip, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("data/knowledge_bases/")

    # Remove the zip file after extraction
    os.remove(output_zip)

    # Remove __MACOSX folder if it exists
    macosx_folder = "data/knowledge_bases/__MACOSX"
    if os.path.exists(macosx_folder):
        shutil.rmtree(macosx_folder)

    print("Files downloaded and extracted successfully.")
else:
    print("All files are present. Proceeding with loading.")

#########
# Model #
#########

# Load HuggingFace API key
with open("api_keys.json", "r") as file:
    hf_access_key = json.load(file).get("HF_ACCESS_KEY")

# Login to HuggingFace
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

# Load FAISS index
faiss_index = faiss.read_index("data/knowledge_bases/poetry_faiss.index")

# Load keyword data
with open('data/knowledge_bases/poetry_unique_keywords.pkl', 'rb') as f:
    unique_keywords_list = pickle.load(f)
    unique_keywords_list = np.array(unique_keywords_list)
with open('data/knowledge_bases/poetry_forward_mapping.pkl', 'rb') as f:
    forward_mapping = pickle.load(f)
with open('data/knowledge_bases/poetry_inverse_mapping.pkl', 'rb') as f:
    inverse_mapping = pickle.load(f)

# Load NetworkX graph
with open("data/knowledge_bases/poetry_keyword_graph.gpickle", "rb") as f:
    pk_graph = pickle.load(f)

# Prepare spacy
nlp = spacy.load('en_core_web_sm')


def format_poems(idx, include_keywords=True, include_tags=False):
    """Format poem text from DataFrame index, removing leading and trailing spaces and newlines."""
    if hasattr(idx, "__len__"):
        return [format_poems(i, include_keywords, include_tags) for i in idx]

    # Strip leading and trailing spaces and newlines
    it = df.iloc[idx]
    title = it["Title"].strip()
    poet = it["Poet"].strip()
    poem = it["Poem"].strip()

    # Construct formatted poem string
    res = f'{title}\n{poet}\n\n{poem}'

    # Optionally include keywords
    if include_keywords:
        res += f'\n\nKeywords: {unique_keywords_list[forward_mapping[idx]].tolist()}'

    # Optionally include tags
    if include_tags:
        res += f'\n\nNotes: {it["Tags"].strip()}'

    return res


###################
# Text Generation #
###################

def generate_kernel(text, temperature=0.1, max_new_tokens=25):
    """Generate text based on input prompt."""
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
    """Endpoint for text generation from a prompt."""
    data = request.json
    text = data.get("text", "")
    temperature = data.get("temperature", 0.1)
    max_new_tokens = data.get("max_new_tokens", 50)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate and return text
    generated_text = generate_kernel(text, temperature=temperature, max_new_tokens=max_new_tokens)
    return jsonify({"generated_text": generated_text})


##############################
# Similarity-based Retrieval #
##############################

def retrieve_faiss_kernel(text, k=1, flatten=True):
    """Retrieve poems similar to the input text using FAISS index."""
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

    # Convert to text format
    poems = format_poems(indices[0], include_keywords=True)
    if flatten:
        poems = "\n\n----------------\n\n".join(poems)
    return poems


@app.route("/retrieve_faiss", methods=["POST"])
def retrieve_faiss():
    """Endpoint for similarity-based poem retrieval."""
    data = request.json
    text = data.get("text", "")
    k = data.get("k", 1)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Compute and return poems
    return jsonify({"retrieved_poems": retrieve_faiss_kernel(text, k, flatten=True)})


###########################
# Keyword-based Retrieval #
###########################

# Template for keyword extraction prompt, asking the model to identify key nouns or verbs
# from user input to perform a graph-based search in the NetworkX keyword graph.
prompt_template = (
    "Identify or infer up to 10 semantically meaningful keywords from the following conversation. "
    "Prioritize information from the most recent conversation turns. "
    "The keywords should be commonly used nouns or verbs. "
    "Provide the keywords directly after `YOUR ANSWER:`, "
    "formatted within brackets and separated by commas, such as "
    "YOUR ANSWER: [teacher, classroom].\n"
    "\n\n%s\n\n\n"
    "YOUR ANSWER: ["
)


def extract_keywords(text):
    """Extract keywords from a given text."""
    inputs = tokenizer(prompt_template % text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].cuda(),
            max_new_tokens=25, temperature=0.1,
            attention_mask=inputs['attention_mask'].cuda(),
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("YOUR ANSWER:")[-1].strip()
    first_left_index = generated_text.find("[")
    first_right_index = generated_text.find("]")
    if first_left_index == -1 or first_right_index == -1:
        return []
    keywords_text = generated_text[first_left_index + 1:first_right_index]
    keywords = [keyword.strip() for keyword in keywords_text.split(",") if keyword.strip()]
    filtered_keywords = []
    for keyword in keywords:
        # Process each keyword individually
        doc = nlp(keyword)
        if doc[0].pos_ in ['NOUN', 'PROPN', 'VERB']:  # filter
            filtered_keywords.append(doc[0].lemma_)  # lemmatize
    return filtered_keywords


def retrieve_poem_ids(query_keywords, k=1, depth=2, depth_decay=0.5):
    """Retrieve most relevant poems based on keywords from NetworkX graph."""
    poem_scores = defaultdict(float)  # Dictionary to accumulate scores for each poem

    for keyword in query_keywords:
        if keyword not in pk_graph:
            continue  # Skip keywords not present in the graph

        # Perform BFS from the keyword node
        queue = deque([(keyword, 1)])  # (current_node, current_depth)
        visited = {keyword}

        while queue:
            current_node, current_depth = queue.popleft()
            if current_depth > depth:
                continue

            # Check if the current node is a poem node (type int) and accumulate score
            if isinstance(current_node, int):
                decay = depth_decay ** (current_depth - 1)  # Decay based on depth
                poem_scores[current_node] += decay

            # Explore neighbors
            for neighbor in pk_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

    # Sort poems by score in descending order and return the top-k poems
    top_k_poems = sorted(poem_scores, key=poem_scores.get, reverse=True)[:k]
    return top_k_poems


def retrieve_nx_graph_kernel(text, k=1, depth=2, depth_decay=0.5, flatten=True):
    """Retrieve poems from NetworkX graph based on keywords in input text."""
    # Extract keywords from query
    query_keywords = extract_keywords(text)
    poems = ""

    if query_keywords:
        # Find most relevant poems based on keyword graph traversal
        indices = retrieve_poem_ids(query_keywords, k, depth, depth_decay)

        # Convert to text format if poems are found
        if indices:
            poems = format_poems(indices, include_keywords=True)
            if flatten:
                poems = "\n\n----------------\n\n".join(poems)

    # Fallback message if no poems were found or if no keywords were extracted
    if not poems:
        poems = "No retrieval results found."

    return poems


@app.route("/retrieve_nx_graph", methods=["POST"])
def retrieve_nx_graph():
    """Endpoint for keyword-based retrieval from the NetworkX graph."""
    data = request.json
    text = data.get("text", "")
    k = data.get("k", 1)
    depth = data.get("depth", 2)
    depth_decay = data.get("depth_decay", 0.5)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Compute and return poems
    return jsonify({"retrieved_poems": retrieve_nx_graph_kernel(text, k, depth, depth_decay, flatten=True)})


# To run the Flask server, set the host to 0.0.0.0 to make it externally accessible
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777)
