import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import logging
from typing import List, Dict
import queue


class DocumentRetriever:
    """
    Class for retrieving relevant documents based on similarity to query.
    Uses threading for parallel computation of similarities.
    """

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        chunks: List[Dict[str, str]],
        num_threads: int = 4,
    ):
        """
        Initialize the DocumentRetriever with document embeddings.

        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary mapping chunk IDs to embedding vectors.
            chunks (List[Dict[str, str]]): List of dictionaries containing chunk ID and text.
            num_threads (int): Number of threads to use for parallel computation.
        """
        self.embeddings = embeddings
        self.chunks = {chunk["id"]: chunk for chunk in chunks}
        self.num_threads = num_threads
        self.logger = logging.getLogger(__name__)

    def _compute_similarities_thread(
        self,
        chunk_ids: List[str],
        query_embedding: np.ndarray,
        result_queue: queue.Queue,
    ) -> None:
        """
        Compute cosine similarities between query embedding and chunk embeddings.

        Args:
            chunk_ids (List[str]): List of chunk IDs to process.
            query_embedding (np.ndarray): Query embedding vector.
            result_queue (queue.Queue): Queue to store results.
        """
        similarities = []

        for chunk_id in chunk_ids:
            if chunk_id in self.embeddings:
                chunk_embedding = self.embeddings[chunk_id]
                # Reshape embeddings for cosine_similarity function
                query_reshaped = query_embedding.reshape(1, -1)
                chunk_reshaped = chunk_embedding.reshape(1, -1)

                # Compute cosine similarity
                similarity = cosine_similarity(query_reshaped, chunk_reshaped)[0][0]
                similarities.append((chunk_id, similarity))

        # Put results in the queue
        result_queue.put(similarities)

    def retrieve_documents(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, any]]:
        """
        Retrieve the top-k most relevant documents based on similarity to query.

        Args:
            query_embedding (np.ndarray): Query embedding vector.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[Dict[str, any]]: List of dictionaries containing document information and similarity scores.
        """
        if not self.embeddings:
            self.logger.error("No embeddings available for retrieval.")
            return []

        # Split chunk IDs into batches for threading
        chunk_ids = list(self.embeddings.keys())
        batch_size = max(1, len(chunk_ids) // self.num_threads)
        batches = [
            chunk_ids[i : i + batch_size] for i in range(0, len(chunk_ids), batch_size)
        ]

        # Create threads and queue for results
        threads = []
        result_queue = queue.Queue()

        for batch in batches:
            thread = threading.Thread(
                target=self._compute_similarities_thread,
                args=(batch, query_embedding, result_queue),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results from queue
        all_similarities = []
        while not result_queue.empty():
            all_similarities.extend(result_queue.get())

        # Sort by similarity score (descending)
        all_similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        top_results = all_similarities[:top_k]

        # Format results
        results = []
        for chunk_id, similarity in top_results:
            if chunk_id in self.chunks:
                results.append(
                    {
                        "id": chunk_id,
                        "text": self.chunks[chunk_id]["text"],
                        "similarity": similarity,
                    }
                )

        self.logger.info(f"Retrieved {len(results)} documents")
        return results
