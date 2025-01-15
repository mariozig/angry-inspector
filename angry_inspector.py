# Standard library imports
import argparse
import logging
from typing import Optional

# Local imports
from document_loader import DocumentLoader
from prompt_generator import PromptGenerator
from query_executor import QueryExecutor
from text_splitter import DocumentSplitter
from vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AngryInspector:
    """
    Main interface for the building code RAG system
    """
    
    def __init__(self):
        """Initialize the inspector with vector store"""
        logger.debug("Initializing AngryInspector")
        self.data_dir = "data"  # Default data directory
        self.vector_store = VectorStore()
        self.prompt_generator = PromptGenerator()
        
    def create_vector_database(self):
        """Create the vector database from documents"""
        try:
            logger.info("Starting vector database creation process...")
            
            # Load documents from data directory
            loader = DocumentLoader(self.data_dir)
            documents = loader.load_documents()
            
            if not documents:
                logger.warning("No documents found in data directory. Please add files before creating the database.")
                return
                
            logger.info(f"Successfully loaded {len(documents)} documents from data directory")
            
            # Split documents into chunks
            splitter = DocumentSplitter()
            chunks = splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks for processing")
            
            # # Clear existing database and add new chunks
            # logger.info("Clearing existing vector database before adding new documents...")
            # self.vector_store.clear_existing_db()
            
            logger.info("Adding documents to vector store...")
            success = self.vector_store.add_documents(chunks)
            
            if success:
                logger.info("Successfully created vector database with all documents")
            else:
                logger.error("Failed to add some documents to the vector store. The database may be incomplete.")
                
        except Exception as e:
            logger.exception("Failed to create vector database")
            
    def query_openai(self, query: str) -> Optional[str]:
        """
        Full RAG pipeline: search -> generate prompt -> query OpenAI
        
        Args:
            query (str): User's question about building codes
            
        Returns:
            Optional[str]: OpenAI's response or None if there's an error
        """
        try:
            logger.info(f'Processing query: "{query}"')
            
            # Search vector database
            logger.debug("Performing similarity search in vector database")
            results = self.vector_store.similarity_search(query)
            
            if not results:
                msg = "No relevant information found. Please try rephrasing your question or being more specific."
                logger.warning(f"No results found for query: {query}")
                return msg
            
            # Generate prompt from search results
            logger.debug("Generating prompt from search results")
            prompt = self.prompt_generator.run(query, results)
            
            # Query OpenAI
            logger.debug("Sending prompt to OpenAI")
            executor = QueryExecutor(prompt)
            response = executor.query_openai()
            
            if not response:
                msg = "An error occurred while analyzing the building codes. Please try again."
                logger.error("Received empty response from OpenAI")
                return msg
                
            logger.info("Successfully generated response from OpenAI")
            return response
            
        except Exception as e:
            logger.exception("Failed to process query through RAG pipeline")
            return None


if __name__ == "__main__":
    def non_empty_string(value):
        if not value or not value.strip():
            raise argparse.ArgumentTypeError("Query cannot be empty")
        return value.strip()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Query building codes using RAG')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--create-db', action='store_true', help='Create/recreate the vector database')
    group.add_argument('--query', type=non_empty_string, help='Search query (cannot be empty)')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = AngryInspector()
    
    # Either create database or perform search
    if args.create_db:
        logger.info("Starting database creation process...")
        inspector.create_vector_database()
    else:
        logger.info(f'Analyzing building codes for query: "{args.query}"')
        response = inspector.query_openai(args.query)
        if response:
            logger.info("Response:\n%s", response)
        else:
            logger.error("Failed to get a response. Please try again.")
