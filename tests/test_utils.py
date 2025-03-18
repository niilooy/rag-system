import pytest
import os
import logging
import tempfile
from unittest.mock import patch, MagicMock
from src.utils import setup_logging, format_time


def test_format_time_microseconds():
    """Test format_time with microsecond values."""
    assert format_time(0.0000125) == "12.50 µs"
    assert format_time(0.0005) == "500.00 µs"
    assert format_time(0.000999) == "999.00 µs"


def test_format_time_milliseconds():
    """Test format_time with millisecond values."""
    assert format_time(0.001) == "1.00 ms"
    assert format_time(0.01) == "10.00 ms"
    assert format_time(0.1) == "100.00 ms"
    assert format_time(0.999) == "999.00 ms"


def test_format_time_seconds():
    """Test format_time with second values."""
    assert format_time(1) == "1.00 s"
    assert format_time(1.5) == "1.50 s"
    assert format_time(10) == "10.00 s"
    assert format_time(60) == "60.00 s"


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_setup_logging_valid_levels(log_level):
    """Test setup_logging with valid log levels."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("utils.os.makedirs") as mock_makedirs:
            with patch("utils.logging.basicConfig") as mock_basicConfig:
                # Change working directory to temp dir
                current_dir = os.getcwd()
                os.chdir(temp_dir)

                # Run the function
                setup_logging(log_level)

                # Verify the logs directory was created
                mock_makedirs.assert_called_once_with("logs", exist_ok=True)

                # Verify basicConfig was called with correct level
                numeric_level = getattr(logging, log_level.upper())
                mock_basicConfig.assert_called_once()
                call_args = mock_basicConfig.call_args[1]
                assert call_args["level"] == numeric_level

                # Return to original directory
                os.chdir(current_dir)


def test_setup_logging_invalid_level():
    """Test setup_logging with an invalid log level."""
    with pytest.raises(ValueError):
        setup_logging("INVALID_LEVEL")


def test_setup_logging_creates_directory():
    """Test that setup_logging creates the logs directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change working directory to temp dir
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        # Run the function
        setup_logging()

        # Verify the logs directory was created
        assert os.path.exists("logs")
        assert os.path.isdir("logs")

        # Return to original directory
        os.chdir(current_dir)


def test_setup_logging_file_creation():
    """Test that setup_logging creates a log file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change working directory to temp dir
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        # Patch datetime to get a consistent filename
        with patch("utils.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250318_120000"

            # Run the function
            setup_logging()

            # Verify the log file was created in the logs directory
            log_path = os.path.join("logs", "rag_system_20250318_120000.log")
            assert os.path.exists(log_path)

        # Return to original directory
        os.chdir(current_dir)
