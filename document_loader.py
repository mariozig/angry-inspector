from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os

class DocumentLoader:
    def __init__(self, directory_path: str):
        """
        Initialize the DocumentLoader
        
        Args:
            directory_path (str): Path to the directory containing documents
        """
        self.directory_path = directory_path

    def _get_loader(self, file_path: str):
        """
        Returns an instance of the appropriate loader based on file extension
        
        Args:
            file_path (str): Path to the file to load
            
        Returns:
            Union[TextLoader, PyPDFLoader]: An instance of the appropriate document loader
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        return TextLoader(file_path)

    def load_documents(self):
        """
        Load documents from the specified directory using LangChain's document loaders
        
        Returns:
            list[langchain_core.documents.base.Document]: List of Document objects, each containing text content and metadata
        """
        documents = []
        
        # Walk through the directory
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    loader = self._get_loader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Successfully loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    continue
        
        print(f"\nLoaded {len(documents)} documents in total from {self.directory_path}")
        return documents

    def run(self):
        """
        Execute the document loading process and display information about loaded documents
        """
        # Ensure the directory exists
        if not os.path.exists(self.directory_path):
            print(f"Directory {self.directory_path} does not exist!")
            return
        
        # Load the documents
        docs = self.load_documents()
        
        # Print some basic information about the loaded documents
        for i, doc in enumerate(docs[:5], 1):  # Print first 5 documents info
            print(f"\nDocument {i}:")
            print(f"Object type: {type(doc)}")
            print(f"Metadata: {doc.metadata}")
            print(f"Content preview: {doc.page_content[:100]}...")

if __name__ == "__main__":
    loader = DocumentLoader("data/ca-burlingame")
    loader.run()
