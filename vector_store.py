from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Dict, Any, Sequence
import os
from dotenv import load_dotenv
import openai
import logging
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class OpenAIError(Exception):
    """Custom exception for OpenAI-related errors"""
    pass

class VectorStore:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize the vector store with Chroma DB using OpenAI embeddings
        
        Args:
            persist_directory (str): Directory to persist the database
            collection_name (Optional[str]): Name of the collection in Chroma DB
            openai_api_key (Optional[str]): OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable
            batch_size (int): Number of documents to process in each batch for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name or "document_collection"
        self.batch_size = batch_size
        
        try:
            # Initialize the embedding function with retry logic
            self.embedding_function = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-ada-002",
                max_retries=5,
                request_timeout=30
            )
            
            # Test the embeddings with a simple query
            self._test_embeddings()
            
        except openai.RateLimitError:
            raise OpenAIError("OpenAI API rate limit exceeded. Please try again later.")
        except openai.AuthenticationError:
            raise OpenAIError("Invalid OpenAI API key. Please check your credentials.")
        except openai.InsufficientQuotaError:
            raise OpenAIError("OpenAI API quota exceeded. Please check your billing settings.")
        except Exception as e:
            raise OpenAIError(f"Error initializing OpenAI embeddings: {str(e)}")
        
        # Initialize Chroma store
        self._initialize_store()
    
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _test_embeddings(self):
        """Test the embedding function with retry logic"""
        self.embedding_function.embed_query("test")
        logger.info("Successfully tested OpenAI embeddings")
    
    def _initialize_store(self):
        """Initialize or load the Chroma database"""
        try:
            self.store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            logger.info(f"Successfully initialized Chroma DB at {self.persist_directory}")
        except Exception as e:
            raise Exception(f"Error initializing Chroma DB: {str(e)}")
    
    def _batch_documents(self, documents: List[Document]) -> List[List[Document]]:
        """Split documents into batches"""
        return [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store in batches
        
        Args:
            documents (List[Document]): List of documents to add
            
        Returns:
            bool: True if documents were added successfully
        """
        try:
            total_batches = len(documents) // self.batch_size + (1 if len(documents) % self.batch_size else 0)
            
            for i, batch in enumerate(self._batch_documents(documents), 1):
                logger.info(f"Processing batch {i}/{total_batches} ({len(batch)} documents)")
                
                # Add retry logic for each batch
                self._add_batch_with_retry(batch)
                
                # Add a small delay between batches to avoid rate limits
                if i < total_batches:
                    time.sleep(1)
            
            self.store.persist()
            logger.info(f"Successfully added {len(documents)} documents to Chroma DB")
            return True
            
        except openai.RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            return False
        except openai.InsufficientQuotaError:
            logger.error("OpenAI API quota exceeded")
            return False
        except Exception as e:
            logger.error(f"Error adding documents to Chroma DB: {str(e)}")
            return False
    
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _add_batch_with_retry(self, batch: List[Document]):
        """Add a batch of documents with retry logic"""
        self.store.add_documents(batch)
    
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents with retry logic
        
        Args:
            query (str): Query text
            k (int): Number of documents to return
            filter (Optional[dict]): Metadata filter
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            results = self.store.similarity_search(
                query,
                k=k,
                filter=filter
            )
            return results
        except openai.RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            return []
        except openai.InsufficientQuotaError:
            logger.error("OpenAI API quota exceeded")
            return []
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            dict: Collection statistics
        """
        try:
            count = self.store._collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": "text-embedding-ada-002"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentLoader
    from text_splitter import DocumentSplitter
    import os
    
    try:
        # Initialize vector store with batch processing
        vector_store = VectorStore(
            persist_directory="chroma_db",
            collection_name="burlingame_codes",
            batch_size=50  # Process 50 documents at a time
        )
        
        # Load and split documents
        loader = DocumentLoader("data/ca-burlingame")
        docs = loader.load_documents()
        
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(docs)
        
        # Add documents to the store
        if vector_store.add_documents(chunks):
            # Print collection stats
            stats = vector_store.get_collection_stats()
            print("\nCollection Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
            
            # Try a sample search
            print("\nSample Search Results:")
            query = "What are the penalties for code violations?"
            results = vector_store.similarity_search(query, k=2)
            
            if results:
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print("Content:", doc.page_content)
                    print("Source:", doc.metadata.get('source'))
                    print("Page:", doc.metadata.get('page'))
            else:
                print("No results found or error occurred during search")
        else:
            print("Failed to add documents to the vector store")
            
    except OpenAIError as e:
        print(f"OpenAI Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
