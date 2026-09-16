

import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

from pdf_vector import INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TOP_K = 4                          # number of chunks to retrieve per question
OLLAMA_MODEL = "llama3"            # free local LLM served by Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# QuestionVectorizer: loads saved index, embeds questions, retrieves + answers
# ---------------------------------------------------------
class QuestionVectorizer:
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []

    def load(self, index_file=INDEX_FILE, chunks_file=CHUNKS_FILE):
        """Load the previously saved FAISS index and chunk metadata."""
        self.index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
        return self

    def embed_question(self, query):
        """Turn a question into a vector using the same embedding model
        used for the PDF chunks -- this is required so they can be compared."""
        return self.model.encode([query]).astype("float32")

    def retrieve(self, query, k=TOP_K):
        """Find the k chunks whose vectors are closest to the question's vector."""
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
        """Full question-answering flow: retrieve -> build prompt -> generate."""
        results = self.retrieve(query)
        prompt = self.build_prompt(query, results)
        return self.call_local_llm(prompt)


# ---------------------------------------------------------
# Interactive Q&A loop
# ---------------------------------------------------------
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