# PDF Parser & RAG Pipeline

A complete **Retrieval-Augmented Generation (RAG)** system that ingests documents (PDFs and text files), creates embeddings, and enables intelligent question-answering using vector search and language models.

## 🎯 Overview

This project implements a full RAG pipeline for document-based question-answering:

1. **Data Ingestion** — Load PDFs and text files
2. **Text Processing** — Chunk documents into manageable pieces
3. **Embeddings** — Generate vector representations using Sentence Transformers
4. **Vector Database** — Store embeddings in ChromaDB for fast retrieval
5. **RAG Retriever** — Retrieve relevant documents based on query similarity
6. **Question Answering** — Generate answers using HuggingFace LLMs (Mistral/Zephyr)

## 📁 Project Structure

```
pdf-parser/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                              # Data directory
│   ├── text_files/                    # Text documents
│   │   ├── AI_intro.txt
│   │   └── python_intro.txt
│   ├── pdf/                           # PDF documents
│   └── vector_store/                  # ChromaDB persistence
│       ├── chroma.sqlite3
│       └── [collection-ids]/
├── notebook/                          # Jupyter notebooks
│   ├── rag_pipeline_complete.ipynb    # Complete RAG implementation
│   ├── document.ipynb                 # Additional demos
│   └── data/                          # Notebook-specific data
│       ├── pdf/
│       ├── text_files/
│       ├── vector_store/
│       └── vector_store_pdf/
└── venv/                              # Python virtual environment
```

## 🛠️ Requirements

- Python 3.8+
- Virtual environment (recommended)

## 📦 Dependencies

- **LangChain** — LLM orchestration framework
- **pypdf & PyMuPDF** — PDF parsing
- **Sentence Transformers** — Embedding generation
- **ChromaDB** — Vector database
- **FAISS** — Vector similarity search
- **HuggingFace** — Free LLM models

All dependencies are listed in `requirements.txt`.

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Clone/navigate to the project
cd pdf-parser

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

Open and run the notebooks in order:

```bash
# Start Jupyter
jupyter notebook

# Open notebook/rag_pipeline_complete.ipynb
```

Or use Jupyter in VS Code for interactive development.

## 📖 Pipeline Components

### Data Ingestion
- Load text files from `data/text_files/` using LangChain's `TextLoader`
- Parse PDFs using `PyPDF` and `PyMuPDF` loaders
- Create `Document` objects with metadata (source, page, author, etc.)

### Text Chunking
- Split documents into semantic chunks (e.g., 1000 tokens with 200-token overlap)
- Preserves context between chunks for better retrieval

### Embeddings
Uses **Sentence Transformers** to convert text into dense vectors:
```python
from sentence_transformers import SentenceTransformer
embeddings = model.encode(texts)
```

### Vector Storage
**ChromaDB** manages embeddings with:
- Persistent storage in `data/vector_store/`
- Fast similarity search
- Metadata filtering
- Collection-based organization

### RAG Retriever
Retrieves most relevant chunks for a query using cosine similarity:
```python
results = vector_store.query(query_embedding, k=5)
```

### Question Answering
Integrates HuggingFace LLMs (Mistral, Zephyr) for free:
- Builds context from retrieved documents
- Generates answers with relevant citations

## 💻 Usage Example

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load documents
loader = DirectoryLoader(
    "data/text_files/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

# Generate embeddings and store
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in documents])

# Store in vector DB
vector_store = VectorStore(collection_name="my_docs")
vector_store.add_documents(documents, embeddings)

# Query
results = vector_store.query(query_embedding, k=5)
```

## 🔍 Key Features

- ✅ **Multi-format support** — PDFs and text files
- ✅ **Semantic search** — Find relevant content by meaning, not keywords
- ✅ **Persistent storage** — ChromaDB saves embeddings across sessions
- ✅ **Modular design** — Easily swap embeddings, LLMs, or vector stores
- ✅ **Free LLMs** — Uses HuggingFace models (no API keys required)
- ✅ **Metadata filtering** — Filter results by source, author, date, etc.

## 📚 Notebooks

- **rag_pipeline_complete.ipynb** — Full implementation with all components
- **document.ipynb** — Additional examples and demos

## 🗂️ Data Formats

### Text Files
Place `.txt` files in `data/text_files/` for automatic loading.

### PDFs
Place `.pdf` files in `data/pdf/` for processing.

## 🔧 Customization

### Change Embedding Model
```python
embedding_manager = EmbeddingManager(model_name="all-MiniLM-L12-v2")
```

### Change LLM
```python
# Switch between Mistral, Zephyr, or other HuggingFace models
llm = HuggingFacePipeline(model_id="mistralai/Mistral-7B-Instruct-v0.1")
```

### Adjust Chunk Size
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)
```

## 📊 Performance Tips

- Use smaller embedding models (e.g., `all-MiniLM-L12-v2`) for faster inference
- Increase `chunk_overlap` for better context preservation
- Use FAISS for GPU-accelerated similarity search
- Cache embeddings in ChromaDB to avoid recomputation

## 🐛 Troubleshooting

**Issue: ChromaDB connection error**
```bash
# Delete corrupted database and restart
rm -rf data/vector_store/chroma.sqlite3
```

**Issue: Out of memory with large PDFs**
- Reduce chunk size
- Process documents in batches

**Issue: Poor RAG results**
- Increase `k` (number of retrieved chunks)
- Fine-tune chunk size and overlap
- Try a different embedding model

## 📝 License

This project is open source and available for personal and educational use.

## 👤 Author

Created by **minahil** — April 2025

---

**Happy RAG-ing! 🚀**
