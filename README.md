# Multi-Threaded Text Retrieval and Processing System

This project implements a lightweight, multi-threaded system in Python that simulates a simplified Retrieval-Augmented Generation (RAG) pipeline. The system extracts and processes data from Wikipedia, creates embeddings, retrieves relevant content based on a query, and processes it efficiently using various parallel programming concepts.

## Project Structure

```
.
├── src/
│   ├── data_extraction.py   # Wikipedia text extraction and cleaning
│   ├── embedding_creation.py   # Multiprocessing-based embeddings creation
│   ├── document_retrieval.py   # Threaded document retrieval functionality
│   ├── text_processing.py   # Async text processing functionality
│   ├── utils.py   # Utility functions for logging and time formatting
│   ├── main.py   # Main script that orchestrates the pipeline
├── tests/
│   ├── test_data_extraction.py
│   ├── test_embedding_creation.py
│   ├── test_document_retrieval.py
│   ├── test_text_processing.py
│   └── test_utils.py
├── logs/ # Directory for logs and output files
├── setup.sh   # Bash script to set up environment and run the program
└── README.md   # This documentation file
└── requirements.txt   # List of dependencies
└── .gitignore   # Gitignore file to tell git what files and directories to ignore while commiting
└── .pylintrc   # Configuration file for pylint
└── pylint_report.txt   # Pylint code quality report
```

## Features

- **Data Extraction and Cleaning (OOP)**: Uses `requests` and `BeautifulSoup` to extract and clean text from Wikipedia pages.
- **Embedding Creation (Multiprocessing)**: Leverages Python's `multiprocessing` module to compute embeddings for text chunks in parallel.
- **Document Retrieval (Threading)**: Implements multi-threaded retrieval using Python's `threading` module to compute similarity metrics.
- **Text Processing (Async Programming)**: Uses `asyncio` to preprocess retrieved chunks concurrently.
- **Comprehensive Logging**: Detailed logging at each step of the pipeline.
- **Error Handling**: Robust error handling and graceful degradation.

## Requirements

- Python 3.7+
- Required Python packages:
  - `requests`
  - `beautifulsoup4`
  - `gensim`
  - `scikit-learn`
  - `numpy`
  - `nltk`
  - `aiofiles`
  - `pytest`
  - `pytest-asyncio`
  - `pylint`
  - `black`

## Installation

### Using the Setup Script

1. Make the setup script executable:

   ```bash
   chmod +x setup.sh
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

### Manual Installation

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python src/main.py "What is the impact of AI?"
```

### Advanced Usage

```bash
python src/main.py "Your Query" --url "https://en.wikipedia.org/wiki/Your_Topic" --top_k 5 --log_level DEBUG
```

### Command-line Arguments

- `query`: The query string to search for in the Wikipedia page (required)
- `--url`: URL of the Wikipedia page to extract data from (default: "https://en.wikipedia.org/wiki/Artificial_intelligence")
- `--top_k`: Number of top results to retrieve (default: 3)
- `--log_level`: Logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO)

## Implementation Details

### Data Extraction and Cleaning (OOP)

The `DataExtractor` class in `data_extraction.py` handles the extraction and cleaning of text data from Wikipedia. It uses object-oriented programming principles to encapsulate the extraction logic:

- **Extraction**: Uses `requests` to fetch the HTML content of a Wikipedia page.
- **Cleaning**: Uses `BeautifulSoup` to parse the HTML and extract relevant text content, removing HTML tags, references, and irrelevant sections.
- **Chunking**: Splits the content into manageable chunks (paragraphs and sections).

### Embedding Creation (Multiprocessing)

The `EmbeddingCreator` class in `embedding_creation.py` creates embeddings for text chunks using multiprocessing:

- **Model Loading**: Loads a pre-trained word embedding model from `gensim`.
- **Parallel Processing**: Uses Python's `multiprocessing.Pool` to create embeddings for text chunks in parallel.
- **Embedding Method**: Creates embeddings by averaging word vectors for each chunk.

### Document Retrieval (Threading)

The `DocumentRetriever` class in `document_retrieval.py` implements multi-threaded retrieval:

- **Similarity Computation**: Uses `threading` to compute cosine similarity between query embedding and chunk embeddings in parallel.
- **Thread Coordination**: Uses queues to collect results from multiple threads.
- **Ranking**: Ranks chunks based on similarity scores and returns the top-k most relevant chunks.

### Text Processing (Async Programming)

The `TextProcessor` class in `text_processing.py` uses asynchronous programming for text processing:

- **Asynchronous Functions**: Uses `asyncio` to define asynchronous functions for text processing.
- **Concurrent Processing**: Processes multiple chunks concurrently using `asyncio.gather()`.
- **Text Processing**: Performs tokenization, stopword removal, and other text preprocessing steps.

## Code Quality with Pylint

This project uses Pylint for code quality assurance. The current Pylint score is **7.73/10**, which indicates good code quality with some room for improvement.

### Running Pylint

You can check the code quality score by running:

```bash
pylint src/*.py > pylint_report.txt
```

Detailed report is saved in pylint_report.txt in the root directory.

### Improving Pylint Score

The main areas for improvement include:

- Documentation completeness
- Variable naming consistency
- Function complexity reduction
- Additional error handling

We maintain a balance between code readability and strict adherence to Pylint rules, sometimes prioritizing practical code clarity over reaching a perfect 10/10 score.

## Testing

The project includes unit tests for all components using pytest. The tests are located in the `tests` directory.


## Output

The system outputs:

1. A ranked list of the top k chunks with their relevance scores, printed to the console.
2. Detailed logs saved to the `logs` directory.
3. Results saved to a text file in the `logs` directory.

## Example Output

For the query "What is the impact of AI?", the system will:

1. Extract and clean text from the Wikipedia page on Artificial Intelligence.
2. Create embeddings for each text chunk using multiprocessing.
3. Create an embedding for the query.
4. Retrieve the top 3 most relevant chunks using threading.
5. Process the retrieved chunks asynchronously.
6. Print the results to the console and saves the results and detailed logs.
7. Results and logs are saved inside logs/.

The output will show the top 3 chunks related to the impact of AI, along with their similarity scores.

## Error Handling

The system includes comprehensive error handling at each step:

- Failed web requests
- Empty chunks
- Model loading failures
- Embedding creation errors
- Processing errors

## Note on Use of Generative AI Tools

During the development of this project, generative AI tools were used primarily for:

- Initial project structure planning
- Debugging assistance for specific functions
- Optimization suggestions for multiprocessing and threading implementations
- Creating this README file.
- Adding unit test cases.

## Limitations and Future Improvements

- The current implementation uses a simple word embedding approach. This could be enhanced with more sophisticated embedding techniques.
- The system currently only extracts data from Wikipedia. Support for other sources could be added.
- The similarity calculation could be improved with more advanced metrics or retrieval methods.
- Additional preprocessing and post-processing steps could be added to improve the quality of the retrieved chunks.
- Code coverage and unit test quality could be improved.
