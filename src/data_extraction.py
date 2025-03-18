import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict
import logging


class DataExtractor:
    """
    Class for extracting and cleaning text data from Wikipedia.
    """

    def __init__(self, url: str):
        """
        Initialize the DataExtractor with a URL.

        Args:
            url (str): The URL of the Wikipedia page to extract data from.
        """
        self.url = url
        self.raw_content = None
        self.soup = None
        self.logger = logging.getLogger(__name__)

    def extract_data(self) -> bool:
        """
        Extract the raw HTML content from the Wikipedia page.

        Returns:
            bool: True if extraction was successful, False otherwise.
        """
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            self.raw_content = response.text
            self.soup = BeautifulSoup(self.raw_content, "html.parser")
            return True
        except requests.RequestException as e:
            self.logger.error(f"Error extracting data from {self.url}: {e}")
            return False

    def clean_data(self) -> List[Dict[str, str]]:
        """
        Clean the extracted data by removing HTML tags, references, and irrelevant sections.
        Split the content into manageable chunks.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing chunk ID and text.
        """
        if not self.soup:
            self.logger.error(
                "No content available for cleaning. Please extract data first."
            )
            return []

        # Get the main content div
        content_div = self.soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            self.logger.error("Could not find main content div.")
            return []

        # Extract paragraphs and section headings
        paragraphs = []

        # Process each section
        for section in content_div.find_all(["h2", "h3", "p"]):
            # Skip navigation sections, references, external links, etc.
            if section.find_parent("div", {"class": "toc"}):
                continue

            # Process headings
            if section.name in ["h2", "h3"]:
                heading_text = section.get_text().strip()
                # Skip sections we don't want
                if any(
                    x in heading_text.lower()
                    for x in [
                        "see also",
                        "references",
                        "external links",
                        "further reading",
                    ]
                ):
                    continue

                if (
                    heading_text
                    and not heading_text.startswith("[")
                    and len(heading_text) > 1
                ):
                    paragraphs.append(
                        {"id": f"heading-{len(paragraphs)}", "text": heading_text}
                    )

            # Process paragraphs
            elif section.name == "p":
                text = section.get_text().strip()
                # Clean up references like [1], [2], etc.
                text = re.sub(r"\[\d+\]", "", text)
                # Remove other brackets
                text = re.sub(r"\[.*?\]", "", text)
                # Remove excessive whitespace
                text = re.sub(r"\s+", " ", text).strip()

                if (
                    text and len(text) > 50
                ):  # Only keep paragraphs with substantial content
                    paragraphs.append({"id": f"para-{len(paragraphs)}", "text": text})

        if not paragraphs:
            self.logger.warning("No content was extracted after cleaning.")

        return paragraphs
