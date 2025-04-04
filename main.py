# app_backend.py
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import uvicorn

app = FastAPI()
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def store_text_in_faiss(text_list):
    dimension = 384
    embeddings = embedding_model.encode(text_list).astype('float32')
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_list

def search_faiss(query, index, text_list, top_k=3):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = [(text_list[i], distances[0][j]) for j, i in enumerate(indices[0]) if i < len(text_list)]
    return results

def ask_llama2(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@app.post("/ask")
async def ask_question(file: UploadFile, question: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        extracted_text = extract_text_from_pdf(tmp_path)
        sentences = [s.strip() for s in extracted_text.split("\n") if s.strip()]
        index, stored_texts = store_text_in_faiss(sentences)
        results = search_faiss(question, index, stored_texts)

        best_context = results[0][0] if results else "No relevant context found."
        answer = ask_llama2(question, best_context)

        return JSONResponse({
            "answer": answer,
            "context": best_context
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
