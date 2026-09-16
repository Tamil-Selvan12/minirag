# Mini RAG (Retrieval-Augmented Generation)

This project demonstrates a minimal local RAG system. It processes a PDF document, chunks the text, creates vector embeddings, and stores them in a FAISS index. You can then ask questions about the document, and a local LLM (running via Ollama) will generate answers based on the retrieved context.

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: You need to have [Ollama](https://ollama.com/) installed to run the local LLM.

## Setup and Installation

1. **Install required Python packages**
   Run the following command to install all the required dependencies:
   ```bash
   pip install PyMuPDF numpy faiss-cpu sentence-transformers requests
   ```

2. **Start the local LLM**
   This project uses the `llama3` model by default. Open a separate terminal and run:
   ```bash
   ollama run llama3
   ```
   *Make sure Ollama is running and accessible at `http://localhost:11434`.*

## How to Execute

### Step 1: Vectorize the PDF
First, you need to extract the text from the PDF, create embeddings, and build the FAISS index.

Make sure your PDF is placed in the project folder and update `PDF_PATH` in `pdf_vector.py` if necessary (it is set to use a local file by default). Then run:
```bash
python pdf_vector.py
```
This will generate two files: `pdf_index.faiss` (the vector index) and `pdf_chunks.pkl` (the text chunks).

### Step 2: Ask Questions
Once the index is built, you can start the interactive Q&A session. Run:
```bash
python question_vectorizer.py
```
Type your questions at the `Q:` prompt. The system will retrieve the most relevant chunks from your PDF and use the local LLM to answer. 

Type `exit` or `quit` to stop the program.