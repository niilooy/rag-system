import asyncio
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import logging
from typing import List, Dict, Any
import nltk

nltk.download("punkt")
nltk.download("stopwords")


class TextProcessor:
    """
    Class for asynchronously processing retrieved text chunks.
    """

    def __init__(self):
        """
        Initialize the TextProcessor.
        """
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words("english"))

    async def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text chunk asynchronously.

        Args:
            chunk (Dict[str, Any]): Dictionary containing chunk information.

        Returns:
            Dict[str, Any]: Dictionary with processed chunk information.
        """
        try:
            # Simulate some async processing time
            await asyncio.sleep(0.01)

            # Tokenize text
            tokens = word_tokenize(chunk["text"])

            # Remove stopwords and punctuation
            filtered_tokens = [
                token.lower()
                for token in tokens
                if token.lower() not in self.stop_words
                and re.match(r"^[a-zA-Z]+$", token)
            ]

            # Create processed text
            processed_text = " ".join(filtered_tokens)

            # Create result dictionary
            result = {
                "id": chunk["id"],
                "original_text": chunk["text"],
                "processed_text": processed_text,
                "token_count": len(filtered_tokens),
                "similarity": chunk.get("similarity", 0.0),
            }

            return result
        except Exception as e:
            self.logger.error(
                f"Error processing chunk {chunk.get('id', 'unknown')}: {e}"
            )
            return {"id": chunk.get("id", "unknown"), "error": str(e)}

    async def process_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple text chunks concurrently.

        Args:
            chunks (List[Dict[str, Any]]): List of dictionaries containing chunk information.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with processed chunk information.
        """
        if not chunks:
            self.logger.warning("No chunks provided for processing.")
            return []

        try:
            # Create a list of tasks for asynchronous processing
            tasks = [self.process_chunk(chunk) for chunk in chunks]

            # Execute all tasks concurrently and collect results
            processed_chunks = await asyncio.gather(*tasks)

            self.logger.info(f"Processed {len(processed_chunks)} chunks successfully")
            return processed_chunks
        except Exception as e:
            self.logger.error(f"Error in concurrent chunk processing: {e}")
            return []
