
from flask import Flask, request, jsonify, render_template
import os, fitz, faiss, torch, nltk
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from peft import PeftModel

nltk.download("punkt")

app = Flask(__name__)

def smart_chunk(text, chunk_size=3):
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def load_all_pdfs(folder_path="user_data"):
    chunks = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, file))
            text = ""
            for page in doc:
                text += page.get_text()
            for para in text.split("\n\n"):
                chunks.extend([c.strip() for c in smart_chunk(para) if c.strip()])
    return chunks

knowledge_base = load_all_pdfs()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kb_embeddings = embedder.encode(knowledge_base, convert_to_numpy=True, normalize_embeddings=True)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

def retrieve_chunks(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(q_emb, top_k)
    return [knowledge_base[i] for i in indices[0]]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.float16
)
model.eval()
torch._dynamo.disable()

chat_log = []

def generate_response(context, query):
    conversation = "\n".join(chat_log[-6:])
    prompt = f"You are Tarun Munukuntla. ONLY answer using this context:\n{context}\n\n{conversation}\nPerson: {query}\nTarun:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Tarun:")[-1].strip()

def autocomplete_suggestions(partial):
    rag_chunks = retrieve_chunks(partial)
    context = "\n".join(rag_chunks)
    base_prompt = f"You are Tarun Munukuntla. ONLY answer using this context:\n{context}\n"
    full_prompt = f"{base_prompt}Person: {partial}\nTarun:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=15, temperature=0.7, do_sample=True,
                                 top_k=50, top_p=0.95, num_return_sequences=4,
                                 pad_token_id=tokenizer.pad_token_id)
    suggestions = []
    for out in outputs:
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        suggestion = decoded.split("Tarun:")[-1].strip()
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
    return suggestions[:4]

chat_cache = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    context_chunks = retrieve_chunks(text)
    responses = [generate_response(chunk, text) for chunk in context_chunks]
    chat_cache["last"] = responses
    chat_log.append(f"Person: {text}")
    return jsonify({"options": responses})

@app.route("/choose", methods=["POST"])
def choose():
    idx = request.json["index"]
    response = chat_cache["last"][idx]
    chat_log.append(f"Tarun: {response}")
    return jsonify({"reply": response})

@app.route("/suggest", methods=["POST"])
def suggest():
    partial = request.json["text"]
    suggestions = []
    if partial.strip():
        try:
            suggestions = autocomplete_suggestions(partial)
        except Exception as e:
            print("Autocomplete error:", e)
    return jsonify({"suggestions": suggestions})

@app.route("/submit_custom", methods=["POST"])
def submit_custom():
    text = request.json["text"]
    chat_log.append(f"Tarun: {text}")
    return jsonify({"reply": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
