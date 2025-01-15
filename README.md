# Angry Inspector

A specialized RAG (Retrieval Augmented Generation) system designed for querying and analyzing building codes and municipal regulations. This library makes it easy to load municipal code documents, process them efficiently, and perform semantic search to quickly find relevant building regulations and requirements.

## Key Features

- Efficient loading and processing of municipal code documents (PDF format)
- Smart text chunking optimized for legal/regulatory content
- Persistent vector storage using ChromaDB
- Natural language querying with context-aware responses
- Built on ChromaDB for vector storage and OpenAI for embeddings and LLM capabilities

## Data Directory Structure

The `data` directory is where you should place your municipal code documents. The system is configured to work with PDF files. Each jurisdiction should have its own subdirectory:

```
data/
└── ca-burlingame/
    └── code-of-ordinances.pdf
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

4. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

The main interface to the system is through the `AngryInspector` class. There are two main operations:

### 1. Creating the Vector Database

Before you can search, you need to create the vector database. This only needs to be done once, or when you add new documents:

```bash
python angry_inspector.py --create-db
```

Note: To reset the database, you'll need to manually delete the contents of the `chroma_db` directory.

### 2. Querying the Building Codes

Once the database is created, you can search using natural language queries:

```bash
python angry_inspector.py --query "What are the requirements for obtaining a building permit?"
```

The system will return relevant information from the building codes in a structured format, including:
- Key requirements and regulations
- Related sections and provisions
- Important conditions and exceptions

## Example Queries

- "What are the requirements for obtaining a building permit?"
- "What are the rules regarding noise and construction hours?"
- "What are the height restrictions for residential buildings?"
- "What are the parking requirements for commercial buildings?"
