```mermaid
flowchart TD
    %% Define Styles
    classDef file fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef process fill:#d4e157,stroke:#333,stroke-width:2px;
    classDef model fill:#ffcc80,stroke:#333,stroke-width:2px;
    classDef storage fill:#81d4fa,stroke:#333,stroke-width:2px;
    classDef user fill:#ce93d8,stroke:#333,stroke-width:2px;

    %% Data Ingestion Phase
    subgraph "Phase 1: Data Ingestion (pdf_vector.py)"
        direction TB
        PDF[PDF Document]:::file
        Extract[Text Extraction\n(PyMuPDF)]:::process
        Chunk[Text Chunking]:::process
        Embed1[Sentence Transformer\n(all-MiniLM-L6-v2)]:::model
        
        PDF --> Extract
        Extract --> Chunk
        Chunk --> Embed1
    end

    %% Storage Phase
    subgraph "Phase 2: Storage"
        direction LR
        FAISS[(FAISS Vector Index\npdf_index.faiss)]:::storage
        Pickle[(Pickle Metadata\npdf_chunks.pkl)]:::storage
        
        Embed1 -->|Embeddings| FAISS
        Chunk -->|Text Content| Pickle
    end

    %% Query Phase
    subgraph "Phase 3: Retrieval & Generation (question_vectorizer.py)"
        direction TB
        Query[User Question]:::user
        Embed2[Sentence Transformer\n(all-MiniLM-L6-v2)]:::model
        Retriever[Similarity Search\n(FAISS Retrieval)]:::process
        Prompt[Prompt Builder]:::process
        LLM[Local LLM\n(Ollama: Llama3)]:::model
        Answer[Generated Answer]:::user

        Query --> Embed2
        Embed2 --> Retriever
        FAISS -.->|Vector Comparison| Retriever
        Retriever -->|Top K Indices| Pickle
        Pickle -.->|Top K Text Chunks| Prompt
        Query --> Prompt
        Prompt -->|Context + Query| LLM
        LLM --> Answer
    end
```
