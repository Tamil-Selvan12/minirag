import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

PDF_PATH = os.getenv("PDF_PATH", os.path.join(os.path.dirname(__file__), "Artificial Intelligence, Machine Learning, and Deep Learning.pdf"))
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 50  # Number of overlapping characters between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "pdf_index.faiss"
CHUNKS_FILE = "pdf_chunks.pkl"

# =================================================================================
# text extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()          # reset every iteration
        if text.strip():
            pages.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    return pages
# chunking================================================================
def chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
            start += chunk_size - chunk_overlap
    return chunks
# pdf_vectorization========================================================
class PDFVectorizer:
    def __init__(self, embedding_model=EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []

    def build_from_pdf(self, pdf_path):
        pages = extract_text_from_pdf(pdf_path)
        self.chunks = chunk_pages(pages)
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype(np.float32))

    def save(self, index_file=INDEX_FILE, chunks_file=CHUNKS_FILE):
        faiss.write_index(self.index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
if __name__ == "__main__":
    vectorizer = PDFVectorizer()
    vectorizer.build_from_pdf(PDF_PATH)
    vectorizer.save()