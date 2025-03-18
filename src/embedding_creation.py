import numpy as np
import gensim.downloader as api
from gensim.utils import simple_preprocess
from multiprocessing import Pool, cpu_count
import logging
from typing import List, Dict, Tuple, Optional


class EmbeddingCreator:
    """
    Class for creating embeddings for text chunks using multiprocessing.
    """

    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        """
        Initialize the EmbeddingCreator with a pre-trained word embedding model.

        Args:
            model_name (str): Name of the pre-trained word embedding model to use.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.vector_size = 0

    def load_model(self) -> bool:
        """
        Load the pre-trained word embedding model.

        Returns:
            bool: True if model was loaded successfully, False otherwise.
        """
        try:
            self.logger.info(f"Loading word embedding model: {self.model_name}")
            self.model = api.load(self.model_name)
            self.vector_size = self.model.vector_size
            self.logger.info(
                f"Model loaded successfully. Vector size: {self.vector_size}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading word embedding model: {e}")
            return False

    def _create_embedding_for_chunk(
        self, chunk: Dict[str, str]
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Create an embedding for a single text chunk by averaging word vectors.

        Args:
            chunk (Dict[str, str]): A dictionary containing chunk ID and text.

        Returns:
            Tuple[str, Optional[np.ndarray]]: Tuple of chunk ID and embedding vector.
        """
        if not self.model:
            return chunk["id"], None

        try:
            # Tokenize text into words
            words = simple_preprocess(chunk["text"])

            # Get word vectors for each word and average them
            word_vectors = [self.model[word] for word in words if word in self.model]

            if not word_vectors:
                return chunk["id"], np.zeros(self.vector_size)

            # Average word vectors to create chunk embedding
            embedding = np.mean(word_vectors, axis=0)
            return chunk["id"], embedding

        except Exception as e:
            self.logger.error(f"Error creating embedding for chunk {chunk['id']}: {e}")
            return chunk["id"], None

    def create_embeddings(self, chunks: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for text chunks using multiprocessing.

        Args:
            chunks (List[Dict[str, str]]): List of dictionaries containing chunk ID and text.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping chunk IDs to embedding vectors.
        """
        if not self.model:
            if not self.load_model():
                return {}

        embeddings = {}
        if not chunks:
            self.logger.warning("No chunks provided for embedding creation.")
            return embeddings

        try:
            # Determine number of processes to use
            num_processes = min(cpu_count(), len(chunks))
            self.logger.info(f"Creating embeddings using {num_processes} processes")

            # Create embeddings in parallel
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_embedding_for_chunk, chunks)

            # Convert results to dictionary
            for chunk_id, embedding in results:
                if embedding is not None:
                    embeddings[chunk_id] = embedding

            self.logger.info(f"Created embeddings for {len(embeddings)} chunks")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error in parallel embedding creation: {e}")
            return {}
