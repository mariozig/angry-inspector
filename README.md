# Angry Inspector

A specialized RAG (Retrieval Augmented Generation) system designed for querying and analyzing building codes and municipal regulations. This library makes it easy to load municipal code documents, process them efficiently, and perform semantic search to quickly find relevant building regulations and requirements.

## Key Features

- Efficient loading of building codes and municipal documents
- Smart text chunking optimized for legal/regulatory content
- Vector storage and semantic search tailored for building codes
- Quick retrieval of relevant regulations using natural language queries
- Detailed context tracking to maintain regulatory accuracy
- Built on ChromaDB and OpenAI embeddings for reliable results

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

3. Create your data directory structure and add your PDF documents:
```bash
mkdir -p data/your-jurisdiction
# Add your PDF documents to this directory
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

The main interface to the system is through the `AngryInspector` class. There are two main operations:

### 1. Creating the Vector Database

Before you can search, you need to create the vector database. This only needs to be done once, or when you add new documents:

```bash
./venv/bin/python angry_inspector.py --create-db
```

### 2. Searching Building Codes

Once the database is created, you can search using natural language queries:

```bash
./venv/bin/python angry_inspector.py --query "What are the height restrictions for residential buildings?"
```

The system will return the most relevant sections from your building codes, including:
- The content of the regulation
- Source document and page number
- Location within the document

## Example Use Cases

- Find specific building requirements for your jurisdiction
- Query parking regulations and zoning requirements
- Search for permit requirements and building restrictions
- Analyze code compliance across different sections
- Cross-reference related regulations
