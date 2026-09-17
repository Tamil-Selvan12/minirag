import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

from pdf_vector import INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL

# Config
TOP_K = 4
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"


class QuestionVectorizer:
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []

    def load(self, index_file=INDEX_FILE, chunks_file=CHUNKS_FILE):
        self.index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
        return self

    def embed_question(self, query):
        return self.model.encode([query]).astype("float32")

    def retrieve(self, query, k=TOP_K):
        query_embedding = self.embed_question(query)
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]

    def build_prompt(self, query, results):
        context = "\n\n".join(
            f"[Page {r['page']}]\n{r['text']}" for r in results
        )
        return f"""You are answering questions using ONLY the context below,
extracted from a PDF document. If the answer isn't in the context, say so
clearly instead of guessing. Cite the page number(s) you used.

Context:
{context}

Question: {query}

Answer:"""

    def call_local_llm(self, prompt, model=OLLAMA_MODEL):
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]

    def ask(self, query):
        results = self.retrieve(query)
        prompt = self.build_prompt(query, results)
        return self.call_local_llm(prompt)


def main():
    qv = QuestionVectorizer()
    qv.load()

    print("\nAsk questions about the PDF (type 'exit' to quit)\n")
    while True:
        query = input("Q: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer = qv.ask(query)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()