# Code-RAG

A specialized RAG (Retrieval Augmented Generation) system designed for querying and analyzing building codes and municipal regulations. This library makes it easy to load municipal code documents, process them efficiently, and perform semantic search to quickly find relevant building regulations and requirements.

## Key Features

- Efficient loading of building codes and municipal documents
- Smart text chunking optimized for legal/regulatory content
- Vector storage and semantic search tailored for building codes
- Quick retrieval of relevant regulations using natural language queries
- Detailed context tracking to maintain regulatory accuracy
- Built on ChromaDB and OpenAI embeddings for reliable results

## Example Use Cases

- Find specific building requirements for your jurisdiction
- Query parking regulations and zoning requirements
- Search for permit requirements and building restrictions
- Analyze code compliance across different sections
- Cross-reference related regulations

## Data Directory Structure

The `data` directory is where you should place your municipal code documents. Currently, the system is configured to work with PDF files. Each jurisdiction should have its own subdirectory:

```
data/
├── ca-burlingame/        # Example jurisdiction
│   ├── building-code.pdf
│   └── zoning-code.pdf
├── ca-san-mateo/
│   └── municipal-code.pdf
└── ...
```

While the system primarily supports PDF files, it can also handle text files. Support for other file formats (Word, HTML, etc.) would require adjustments to the `document_loader.py`.

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create your data directory structure:
```bash
mkdir -p data/your-jurisdiction
```

4. Add your PDF documents to the appropriate jurisdiction directory

5. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Loading Documents

The document loader will process all PDF and text files in the specified directory:

```python
from document_loader import DocumentLoader

# Load documents from a specific jurisdiction
loader = DocumentLoader("data/your-jurisdiction")
documents = loader.load_documents()
```

### Processing Documents

```python
from text_splitter import DocumentSplitter
from vector_store import VectorStore

# Split documents into chunks
splitter = DocumentSplitter()
chunks = splitter.split_documents(documents)

# Store in vector database for searching
vector_store = VectorStore()
vector_store.add_documents(chunks)

# Search for specific regulations
results = vector_store.query("What are the height restrictions for residential buildings?")
for doc in results:
    print(doc.page_content)
```

## Features

- Recursive document loading from directories
- Automatic file type detection (PDF, text)
- Configurable text chunking with overlap
- Progress tracking and chunk statistics
- Semantic search using OpenAI embeddings
- Vector storage using ChromaDB for fast retrieval

## Class Usage

### VectorStore

The `VectorStore` class provides vector storage and retrieval using Chroma DB and OpenAI embeddings:

```python
from vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore(
    persist_directory="chroma_db",
    collection_name="my_collection",  # Optional
    batch_size=100  # Optional
)

# Add documents
vector_store.add_documents(documents)

# Query similar documents
results = vector_store.query(
    query_text="your search query",
    n_results=5  # Optional, defaults to 4
)
```

### DocumentSplitter

The `DocumentSplitter` class splits documents into smaller chunks for better processing:

```python
from text_splitter import DocumentSplitter

# Initialize splitter with custom settings
splitter = DocumentSplitter(
    chunk_size=1000,  # Optional
    chunk_overlap=500,  # Optional
    add_start_index=True  # Optional
)

# Split documents
chunks = splitter.split_documents(documents)

# Get chunk statistics
chunk_info = splitter.get_chunk_info(chunks)
print(chunk_info)  # Shows total chunks, average/min/max chunk sizes
```

Note: Make sure to set your OpenAI API key in the environment variables or pass it directly to the VectorStore constructor.
