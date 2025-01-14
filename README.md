# Document Loader

A Python script that uses LangChain to load and process documents from a specified directory.

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

## Usage

The script will load documents from the `data/ca-burlingame` directory:

```bash
python document_loader.py
```

## Features

- Recursively loads all text documents from the specified directory
- Shows loading progress
- Provides basic information about loaded documents
- Uses LangChain's DirectoryLoader for efficient document processing
