import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.text_processing import TextProcessor


@pytest.fixture
def text_processor():
    return TextProcessor()


def test_text_processor_initialization():
    """Test that TextProcessor initializes correctly with stopwords."""
    processor = TextProcessor()
    assert hasattr(processor, "stop_words")
    assert len(processor.stop_words) > 0
    assert "the" in processor.stop_words
    assert "and" in processor.stop_words


@pytest.mark.asyncio
async def test_process_chunk_removes_stopwords():
    """Test that process_chunk correctly removes stopwords from text."""
    processor = TextProcessor()
    chunk = {
        "id": "test-1",
        "text": "This is a test with some stopwords.",
        "similarity": 0.85,
    }

    result = await processor.process_chunk(chunk)

    assert "is" not in result["processed_text"]
    assert "a" not in result["processed_text"]
    assert "with" not in result["processed_text"]
    assert "test" in result["processed_text"]
    assert "stopwords" in result["processed_text"]


@pytest.mark.asyncio
async def test_process_chunk_handles_empty_text():
    """Test that process_chunk handles empty text properly."""
    processor = TextProcessor()
    chunk = {"id": "test-empty", "text": "", "similarity": 0.5}

    result = await processor.process_chunk(chunk)

    assert result["processed_text"] == ""
    assert result["token_count"] == 0
    assert result["original_text"] == ""


@pytest.mark.asyncio
async def test_process_chunk_preserves_metadata():
    """Test that process_chunk preserves original metadata."""
    processor = TextProcessor()
    chunk = {"id": "test-meta", "text": "Keep this metadata intact", "similarity": 0.75}

    result = await processor.process_chunk(chunk)

    assert result["id"] == "test-meta"
    assert result["original_text"] == "Keep this metadata intact"
    assert result["similarity"] == 0.75


@pytest.mark.asyncio
async def test_process_chunk_handles_error():
    """Test that process_chunk handles errors gracefully."""
    processor = TextProcessor()

    # Simulate an error by using a chunk with incorrect structure
    chunk = {"id": "test-error", "wrong_key": "This will cause an error"}

    result = await processor.process_chunk(chunk)

    assert result["id"] == "test-error"
    assert "error" in result


@pytest.mark.asyncio
async def test_process_chunks_handles_multiple_chunks():
    """Test that process_chunks processes multiple chunks correctly."""
    processor = TextProcessor()
    chunks = [
        {"id": "chunk-1", "text": "First chunk with words", "similarity": 0.9},
        {
            "id": "chunk-2",
            "text": "Second chunk with different words",
            "similarity": 0.8,
        },
    ]

    results = await processor.process_chunks(chunks)

    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[1]["id"] == "chunk-2"
    assert "first chunk words" in results[0]["processed_text"]
    assert "second chunk different words" in results[1]["processed_text"]


@pytest.mark.asyncio
async def test_process_chunks_handles_empty_list():
    """Test that process_chunks handles an empty list properly."""
    processor = TextProcessor()

    results = await processor.process_chunks([])

    assert results == []


@pytest.mark.asyncio
async def test_process_chunks_concurrent_execution():
    """Test that process_chunks executes concurrently."""
    processor = TextProcessor()

    # Create many chunks to test concurrency
    chunks = [
        {"id": f"chunk-{i}", "text": f"Text for chunk {i}", "similarity": 0.5}
        for i in range(10)
    ]

    # Mock the process_chunk method to track calls
    original_process_chunk = processor.process_chunk

    # Use a mock to track when process_chunk is called
    call_times = []

    async def mock_process_chunk(chunk):
        call_times.append(asyncio.current_task())
        # Add a small delay to ensure concurrent execution
        await asyncio.sleep(0.01)
        return await original_process_chunk(chunk)

    processor.process_chunk = mock_process_chunk

    # Process all chunks
    results = await processor.process_chunks(chunks)

    # Check that all chunks were processed
    assert len(results) == 10

    # Check that there were multiple tasks (indicating concurrent execution)
    assert len(set(call_times)) > 1
