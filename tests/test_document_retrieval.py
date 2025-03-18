import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import queue
import threading
from src.document_retrieval import DocumentRetriever


class TestDocumentRetriever:
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return {
            "para-0": np.array([0.1, 0.2, 0.3]),
            "para-1": np.array([0.4, 0.5, 0.6]),
            "para-2": np.array([0.7, 0.8, 0.9]),
        }

    @pytest.fixture
    def sample_chunks(self):
        """Sample text chunks for testing."""
        return [
            {"id": "para-0", "text": "This is the first paragraph."},
            {"id": "para-1", "text": "This is the second paragraph."},
            {"id": "para-2", "text": "This is the third paragraph."},
        ]

    @pytest.fixture
    def document_retriever(self, sample_embeddings, sample_chunks):
        """Create a DocumentRetriever instance."""
        return DocumentRetriever(sample_embeddings, sample_chunks)

    def test_init(self, document_retriever, sample_embeddings, sample_chunks):
        """Test initialization of DocumentRetriever."""
        assert document_retriever.embeddings == sample_embeddings
        assert document_retriever.chunks == {
            "para-0": sample_chunks[0],
            "para-1": sample_chunks[1],
            "para-2": sample_chunks[2],
        }
        assert document_retriever.num_threads == 4

    def test_compute_similarities_thread(self, document_retriever):
        """Test similarity computation in a thread."""
        # Setup
        chunk_ids = ["para-0", "para-1"]
        query_embedding = np.array([0.1, 0.2, 0.3])
        result_queue = queue.Queue()

        # Call the method
        document_retriever._compute_similarities_thread(
            chunk_ids, query_embedding, result_queue
        )

        # Get results from queue
        similarities = result_queue.get()

        # Assertions
        assert len(similarities) == 2
        assert similarities[0][0] == "para-0"
        assert similarities[1][0] == "para-1"
        assert isinstance(similarities[0][1], float)
        assert isinstance(similarities[1][1], float)

    def test_retrieve_documents(self, document_retriever):
        """Test document retrieval."""
        # Setup
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Call the method
        results = document_retriever.retrieve_documents(query_embedding, top_k=2)

        # Assertions
        assert len(results) == 2
        assert "id" in results[0]
        assert "text" in results[0]
        assert "similarity" in results[0]
        # The most similar document should be the first one based on our sample data
        assert results[0]["id"] == "para-0"

    def test_retrieve_documents_empty_embeddings(self):
        """Test document retrieval with empty embeddings."""
        # Setup
        retriever = DocumentRetriever({}, [])
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Call the method
        results = retriever.retrieve_documents(query_embedding)

        # Assertions
        assert results == []
