import os
import requests
import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader

# =========================
# Config
# =========================
DOCS_FOLDER = "project_docs"   # Put your PDFs here
MODEL_NAME = "llama3.1:8b"     # Any Ollama model you have pulled
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding model

# =========================
# Setup ChromaDB
# =========================
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="my_rag")

# =========================
# Helper: Get embedding from Ollama
# =========================
def get_ollama_embedding(text: str):
    url = "http://localhost:11434/api/embeddings"
    response = requests.post(url, json={"model": EMBED_MODEL, "prompt": text})
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Ollama embedding error: {response.text}")

# =========================
# Helper: Load PDF
# =========================
def load_pdf_file(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

# =========================
# Load PDFs from folder
# =========================
def load_documents_from_folder(folder):
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    for idx, pdf_file in enumerate(pdf_files):
        path = os.path.join(folder, pdf_file)
        text = load_pdf_file(path)
        if text:
            embedding = get_ollama_embedding(text)
            doc_id = f"doc_{idx}"
            collection.add(documents=[text], embeddings=[embedding], ids=[doc_id])
    return len(pdf_files)

num_loaded = load_documents_from_folder(DOCS_FOLDER)

# =========================
# RAG Query Function
# =========================
def rag_query(question):
    # Get query embedding
    query_embedding = get_ollama_embedding(question)

    # Retrieve relevant docs
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context_docs = results["documents"][0]
    context = "\n".join(context_docs)

    # Ask Ollama with context
    prompt = f"""
    You are a helpful assistant. Use the context below to answer the question.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# =========================
# Streamlit UI
# =========================
st.title("Local RAG Chatbot with Ollama + PDFs")

if num_loaded > 0:
    st.write(f"Loaded {num_loaded} PDF(s) from '{DOCS_FOLDER}'.")
else:
    st.write(f"No PDFs found in folder: {DOCS_FOLDER}")

# Keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the PDFs:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        answer = rag_query(query)
        # Save to history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**{sender}:** {msg}")
    else:
        st.markdown(f"**{sender}:** {msg}")
