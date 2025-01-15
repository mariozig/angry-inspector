# Standard library imports
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

# Third-party imports
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import APIError, AuthenticationError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

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
            clear_db (bool): Whether to clear the existing database on initialization
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name or "document_collection"
        self.batch_size = batch_size
        
        try:
            self.embedding_function = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-3-small",
                max_retries=5,
                request_timeout=30
            )
        except AuthenticationError:
            raise Exception("Invalid OpenAI API key. Please check your credentials.")
        except Exception as e:
            raise Exception(f"Error initializing OpenAI embeddings: {str(e)}")
        
        try:
            client_settings = Settings(
                anonymized_telemetry=False,
                persist_directory=self.persist_directory
            )
            
            self.store = Chroma(
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
                client_settings=client_settings
            )
            logger.info(f"Successfully initialized Chroma DB at {self.persist_directory}")
        except Exception as e:
            raise Exception(f"Error initializing Chroma DB: {str(e)}")
    
    ## TODO: For now you must manually clear the database by deleting everything in the dir
    # def clear_existing_db(self):
    #     """Empty the Chroma database directory if it exists"""
    #     if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
    #         for item in os.listdir(self.persist_directory):
    #             item_path = os.path.join(self.persist_directory, item)
    #             if os.path.isfile(item_path):
    #                 os.unlink(item_path)
    #             elif os.path.isdir(item_path):
    #                 shutil.rmtree(item_path)
    #         logger.info(f"Emptied directory contents at {self.persist_directory}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    def _test_embeddings(self):
        """Test the embedding function with retry logic"""
        self.embedding_function.embed_query("test")
        logger.info("Successfully tested OpenAI embeddings")
    
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
            
            logger.info(f"Successfully added {len(documents)} documents to Chroma DB")
            return True
            
        except RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            return False
        except Exception as e:
            logger.error(f"Error adding documents to Chroma DB: {str(e)}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    def _add_batch_with_retry(self, batch: List[Document]):
        """Add a batch of documents with retry logic"""
        self.store.add_documents(batch)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: float = 0.3
    ) -> List[Document]:
        """
        Search for similar documents with relevancy scores and filtering
        
        Args:
            query (str): Query text
            k (int): Number of documents to return
            filter (Optional[dict]): Metadata filter
            score_threshold (float): Minimum relevancy score threshold (0-1)
            
        Returns:
            List[Document]: List of similar documents that meet the relevancy threshold
        """
        try:
            results = self.store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=filter
            )

            # Filter out results below the threshold
            filtered_results = []
            for doc, score in results:
                if score >= score_threshold:
                    filtered_results.append(doc)
                    
            if not filtered_results:
                logger.info(f"No results found with relevancy score >= {score_threshold}")
                
            return filtered_results
            
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
                "embedding_model": "text-embedding-3-small"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    from document_loader import DocumentLoader
    from text_splitter import DocumentSplitter
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory="chroma_db",
            collection_name="burlingame_codes",
            clear_db=True
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
            query = "What are the requirements for residential building permits?"
            results = vector_store.similarity_search(query, k=2)
            
            if results:
                print(f"\nSearch Results for: {query}")
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print("Content:", doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")
                    print("Source:", doc.metadata.get("source", "N/A"))
                    print("Page:", doc.metadata.get("page", "N/A"))
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
