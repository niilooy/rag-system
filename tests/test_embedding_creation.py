import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from multiprocessing import Pool
from src.embedding_creation import EmbeddingCreator


class TestEmbeddingCreator:
    @pytest.fixture
    def embedding_creator(self):
        """Create an EmbeddingCreator instance."""
        return EmbeddingCreator()

    @pytest.fixture
    def sample_chunks(self):
        """Sample text chunks for testing."""
        return [
            {"id": "para-0", "text": "This is the first paragraph."},
            {"id": "para-1", "text": "This is the second paragraph."},
        ]

    @patch("gensim.downloader.load")
    def test_load_model_success(self, mock_load, embedding_creator):
        """Test successful model loading."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.vector_size = 100
        mock_load.return_value = mock_model

        # Call the method
        result = embedding_creator.load_model()

        # Assertions
        assert result is True
        assert embedding_creator.model == mock_model
        assert embedding_creator.vector_size == 100
        mock_load.assert_called_once_with("glove-wiki-gigaword-100")

    @patch("gensim.downloader.load")
    def test_load_model_failure(self, mock_load, embedding_creator):
        """Test model loading failure."""
        # Setup mock to raise an exception
        mock_load.side_effect = Exception("Model not found")

        # Call the method
        result = embedding_creator.load_model()

        # Assertions
        assert result is False
        assert embedding_creator.model is None
        assert embedding_creator.vector_size == 0

    def test_create_embedding_for_chunk_no_model(self, embedding_creator):
        """Test embedding creation when no model is loaded."""
        # Ensure model is None
        embedding_creator.model = None

        # Call the method
        chunk_id, embedding = embedding_creator._create_embedding_for_chunk(
            {"id": "para-0", "text": "Test text"}
        )

        # Assertions
        assert chunk_id == "para-0"
        assert embedding is None

    @patch("gensim.utils.simple_preprocess")
    def test_create_embedding_for_chunk_success(
        self, mock_simple_preprocess, embedding_creator
    ):
        """Test successful embedding creation for a chunk."""
        # Setup
        mock_model = MagicMock()
        mock_model.vector_size = 3
        mock_model.__getitem__.side_effect = lambda x: np.array([0.1, 0.2, 0.3])
        embedding_creator.model = mock_model
        embedding_creator.vector_size = 3
        mock_simple_preprocess.return_value = ["this", "is", "a", "test"]

        # Call the method
        chunk_id, embedding = embedding_creator._create_embedding_for_chunk(
            {"id": "para-0", "text": "This is a test"}
        )

        # Assertions
        assert chunk_id == "para-0"
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)

    @patch("multiprocessing.Pool")
    def test_create_embeddings(self, mock_pool, embedding_creator, sample_chunks):
        """Test embedding creation for multiple chunks."""
        # Setup
        mock_model = MagicMock()
        mock_model.vector_size = 3
        embedding_creator.model = mock_model

        # Mock the pool.map result
        mock_instance = MagicMock()
        mock_instance.map.return_value = [
            ("para-0", np.array([0.1, 0.2, 0.3])),
            ("para-1", np.array([0.4, 0.5, 0.6])),
        ]
        mock_pool.return_value.__enter__.return_value = mock_instance

        # Call the method
        result = embedding_creator.create_embeddings(sample_chunks)

        # Assertions
        assert len(result) == 2
        assert "para-0" in result
        assert "para-1" in result
        assert isinstance(result["para-0"], np.ndarray)
        assert isinstance(result["para-1"], np.ndarray)
