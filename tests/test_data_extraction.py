import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
import requests
from src.data_extraction import DataExtractor


class TestDataExtractor:
    @pytest.fixture
    def mock_wikipedia_html(self):
        """Sample Wikipedia HTML for testing."""
        return """
        <html>
            <body>
                <div id="mw-content-text">
                    <h2>Introduction</h2>
                    <p>This is a sample paragraph about artificial intelligence. It has more than 50 characters.</p>
                    <h2>History</h2>
                    <p>AI has a long history[1][2] dating back to the 1950s. This text should be long enough to pass the filter.</p>
                    <p>This paragraph is too short.</p>
                    <h2>See also</h2>
                    <p>This should be skipped because it's in a section we don't want.</p>
                    <div class="toc">
                        <h2>Table of Contents</h2>
                        <p>This should be skipped because it's in the TOC.</p>
                    </div>
                </div>
            </body>
        </html>
        """

    @pytest.fixture
    def data_extractor(self):
        """Create a DataExtractor instance."""
        return DataExtractor("https://en.wikipedia.org/wiki/Artificial_intelligence")

    @patch("requests.get")
    def test_extract_data_success(self, mock_get, data_extractor):
        """Test successful data extraction."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "<html>Test</html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the method
        result = data_extractor.extract_data()

        # Assertions
        assert result is True
        assert data_extractor.raw_content == "<html>Test</html>"
        assert isinstance(data_extractor.soup, BeautifulSoup)
        mock_get.assert_called_once_with(
            "https://en.wikipedia.org/wiki/Artificial_intelligence", timeout=10
        )

    @patch("requests.get")
    def test_extract_data_failure(self, mock_get, data_extractor):
        """Test data extraction failure."""
        # Setup mock to raise an exception
        mock_get.side_effect = requests.RequestException("Connection error")

        # Call the method
        result = data_extractor.extract_data()

        # Assertions
        assert result is False
        assert data_extractor.raw_content is None
        assert data_extractor.soup is None

    def test_clean_data_no_soup(self, data_extractor):
        """Test clean_data when no soup is available."""
        # Ensure soup is None
        data_extractor.soup = None

        # Call the method
        result = data_extractor.clean_data()

        # Assertions
        assert result == []

    def test_clean_data_success(self, data_extractor, mock_wikipedia_html):
        """Test successful data cleaning."""
        # Setup soup with mock HTML
        data_extractor.soup = BeautifulSoup(mock_wikipedia_html, "html.parser")

        # Call the method
        result = data_extractor.clean_data()

        # Assertions
        assert len(result) == 3  # Two paragraphs and one heading
        assert result[0]["text"] == "Introduction"
        assert "artificial intelligence" in result[1]["text"]
        assert "AI has a long history" in result[2]["text"]
        assert not any("See also" in item["text"] for item in result)
        assert not any("too short" in item["text"] for item in result)
