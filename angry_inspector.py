from typing import List, Optional

from document_loader import DocumentLoader
from text_splitter import DocumentSplitter
from vector_store import VectorStore
from langchain_core.documents import Document


class AngryInspector:
    def __init__(
        self,
        data_directory: str = "data",
        persist_directory: str = "chroma_db",
        collection_name: str = "building_codes"
    ):
        """
        Initialize AngryInspector with configuration for document processing and vector storage
        
        Args:
            data_directory (str): Directory containing building code documents
            persist_directory (str): Directory to persist the vector database
            collection_name (str): Name of the collection in the vector database
        """
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = None

    def create_vector_database(self) -> bool:
        """
        Create the vector database from documents in the data directory
        
        Returns:
            bool: True if database was created successfully
        """
        try:
            # Initialize components
            loader = DocumentLoader(self.data_directory)
            splitter = DocumentSplitter()
            self.vector_store = VectorStore(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                clear_db=True
            )

            # Load and process documents
            documents = loader.load_documents()
            if not documents:
                print("No documents found in the data directory")
                return False

            # Split documents into chunks
            chunks = splitter.split_documents(documents)
            if not chunks:
                print("Failed to split documents into chunks")
                return False

            # Add to vector store
            success = self.vector_store.add_documents(chunks)
            if success:
                stats = self.vector_store.get_collection_stats()
                print("\nVector Database Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
                return True
            return False

        except Exception as e:
            print(f"Error creating vector database: {str(e)}")
            return False

    def search(
        self,
        query: str,
        num_results: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search the vector database for relevant building code sections
        
        Args:
            query (str): The search query
            num_results (int): Number of results to return
            filter (Optional[dict]): Optional metadata filter
            
        Returns:
            List[Document]: List of relevant document chunks
        """
        if not self.vector_store:
            try:
                self.vector_store = VectorStore(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
            except Exception as e:
                print(f"Error connecting to vector database: {str(e)}")
                return []

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=num_results,
                filter=filter
            )
            
            if not results:
                print("\nNo relevant building codes found for your query. Try rephrasing or being more specific.")
                return []
                
            return results
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            return []


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Query building codes using RAG')
    parser.add_argument('--create-db', action='store_true', help='Create/recreate the vector database')
    parser.add_argument('--query', type=str, help='Search query')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = AngryInspector()
    
    # Handle different operations
    if args.create_db:
        print("\nCreating vector database...")
        inspector.create_vector_database()
    if args.query:
        print(f"\nSearch Results for: {args.query}")
        results = inspector.search(args.query)
        
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("Content:", doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")
            print("Metadata:", doc.metadata)
    if not args.create_db and not args.query:
        print("\nError: Please specify either --create-db to create the database or --query to search")
        parser.print_help()
        exit(1)
