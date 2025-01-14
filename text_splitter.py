from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 500,
        length_function = len,
        add_start_index: bool = True
    ):
        """
        Initialize the DocumentSplitter with configuration for text splitting
        
        Args:
            chunk_size (int): The size of each text chunk
            chunk_overlap (int): The number of characters to overlap between chunks
            length_function (callable): Function to measure text length
            add_start_index (bool): Whether to add start index to metadata
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            add_start_index=add_start_index,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of split documents
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return documents
    
    def get_chunk_info(self, chunks: List[Document]) -> dict:
        """
        Get information about the chunks
        
        Args:
            chunks (List[Document]): List of document chunks
            
        Returns:
            dict: Dictionary containing chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_lengths) / len(chunks),
            "min_chunk_size": min(chunk_lengths),
            "max_chunk_size": max(chunk_lengths),
        }

if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentLoader
    
    # Load some documents
    loader = DocumentLoader("data/ca-burlingame")
    docs = loader.load_documents()
    
    # Create splitter and split documents
    splitter = DocumentSplitter()
    chunks = splitter.split_documents(docs)
    
    # Print chunk information
    info = splitter.get_chunk_info(chunks)
    print("\nChunk Statistics:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
