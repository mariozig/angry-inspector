from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
#
# { context}
#
# ---
#
# Answer the question based on the above context: {question}
# """

class PromptGenerator:
    """
    Generates prompts for the language model based on retrieved documents
    """
    
    def run(self, query: str, documents: List[Document]) -> str:
        """
        Generate a prompt combining the user query and relevant document snippets
        
        Args:
            query (str): The user's original query
            documents (List[Document]): List of relevant documents from vector search
            
        Returns:
            str: Generated prompt for the language model
        """
        try:
            # Start with the user's query
            prompt_parts = [
                "Based on the following building code sections, please answer this question:",
                f"\nQuestion: {query}\n",
                "\nRelevant building codes:"
            ]
            
            # Add each relevant document with its source
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'Unknown page')
                prompt_parts.append(
                    f"\nSection {i} (from {source}, page {page}):"
                    f"\n{doc.page_content}\n"
                )
            
            # Add final instruction
            prompt_parts.append(
                "\nPlease provide a clear and concise answer based on these building codes. "
                "If the codes don't fully address the question, please mention that."
            )
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return ""
