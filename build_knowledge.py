from pathlib import Path
from PyPDF2 import PdfReader
from fastembed import TextEmbedding
import numpy as np
import google.generativeai as genai

genai.configure(api_key="API_KEY")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text+=extracted + "\n"

    return text

def load_documents(folder):
    documents = []
    folder = Path(folder)
    for file in folder.iterdir():
        if file.suffix == '.txt':
            content = load_txt(file)
            documents.append(
                {
                    "source": file.name,
                    "content": content
                }
            )
        elif file.suffix == '.pdf':
            content = load_pdf(file)
            documents.append(
                {
                    "source": file.name,
                    "content": content
                }
            )
    return documents


def split_chunks(documents, chunk_size=500, overlap=100):
    chunks = []
    for doc in documents:
        text = doc["content"]
        start = 0
        while start < len(text):
            end =  start + chunk_size
            piece = text[start:end]
            chunks.append({
                "source": doc["source"],
                "content": piece
            })
            start += chunk_size - overlap
    return chunks


def build_embeddings(chunks):
    
    texts = []
    for chunk in chunks:
        texts.append(chunk["content"])
    vectors = list(embedding_model.embed(texts))
    vectors = np.array(vectors)
    return vectors

def cosine_similarity(query_vector, vectors):
    dot_product = np.dot(vectors, query_vector)
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    similarity = (dot_product/(vector_norms * query_norm))
    return similarity

def search(question, embedding_model, vectors, chunks, top_k=3):
    query_embedding = np.array(list(embedding_model.embed([question]))[0])
    scores = cosine_similarity(query_embedding, vectors)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append((scores[idx], chunks[idx]))
    return results


def generate_answer(question, retrieved_chunks):
    context=""
    for score, chunk in retrieved_chunks:
        context+=(chunk["content"]+"\n\n")

    prompt = f"""
    Answer ONLY from provided context. If information is unavailable,
    say:
    'I could not find that information.'

    Context:
    {context}

    Question:
    {question}

    """
    response = gemini_model.generate_content(prompt)
    return response.text

docs = load_documents("data")

chunks =  split_chunks(docs)

print("Chunks created:", len(chunks))

vectors = build_embeddings(chunks)

while True:
    question = input("\nAsk: ")
    if question.lower() == "exit":
        break
    results = search(question, embedding_model, vectors, chunks)
    answer = generate_answer(question, results)
    print("\nAnswer:\n")
    print(answer)