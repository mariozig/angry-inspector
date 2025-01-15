import os
from typing import Optional
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class QueryExecutor:
    """
    Executes prompts against OpenAI's API
    """
    
    def __init__(self, prompt: str):
        """
        Initialize with a prompt
        
        Args:
            prompt (str): The prompt to send to OpenAI
        """
        self.prompt = prompt
        # API key is infered from environment variable OPENAI_API_KEY
        self.client = OpenAI()
        
    def query_openai(self, model: str = "gpt-4o") -> Optional[str]:
        """
        Execute the prompt against OpenAI's API
        
        Args:
            model (str): The OpenAI model to use
            
        Returns:
            Optional[str]: The model's response or None if there's an error
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful building code expert. Provide clear, accurate answers based on the building codes provided."},
                    {"role": "user", "content": self.prompt}
                ],
                temperature=0,  # Keep responses focused and consistent
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return None
