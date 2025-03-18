#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
import argparse
from datetime import datetime

# Import our modules
from data_extraction import DataExtractor
from embedding_creation import EmbeddingCreator
from document_retrieval import DocumentRetriever
from text_processing import TextProcessor
from utils import setup_logging


async def main():
    """
    Main function that orchestrates the RAG pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Retrieval-Augmented Generation (RAG) Pipeline"
    )
    parser.add_argument(
        "query", type=str, help="Query to search for in the Wikipedia page"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://en.wikipedia.org/wiki/Artificial_intelligence",
        help="URL of the Wikipedia page to extract data from",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting RAG pipeline with query: '{args.query}'")
    logger.info(f"Using URL: {args.url}")

    # Step 1: Extract and clean data from Wikipedia
    logger.info("Step 1: Extracting and cleaning data from Wikipedia...")
    data_extractor = DataExtractor(url=args.url)

    if not data_extractor.extract_data():
        logger.error("Failed to extract data from Wikipedia. Exiting.")
        return 1

    chunks = data_extractor.clean_data()
    if not chunks:
        logger.error("No clean chunks extracted from the data. Exiting.")
        return 1

    logger.info(
        f"Successfully extracted and cleaned {len(chunks)} chunks from Wikipedia."
    )

    # Step 2: Create embeddings for chunks using multiprocessing
    logger.info("Step 2: Creating embeddings for chunks using multiprocessing...")
    embedding_creator = EmbeddingCreator()
    chunk_embeddings = embedding_creator.create_embeddings(chunks)

    if not chunk_embeddings:
        logger.error("Failed to create embeddings. Exiting.")
        return 1

    logger.info(f"Successfully created embeddings for {len(chunk_embeddings)} chunks.")

    # Step 3: Create embedding for the query
    logger.info("Step 3: Creating embedding for the query...")
    query_chunks = [{"id": "query", "text": args.query}]
    query_embeddings = embedding_creator.create_embeddings(query_chunks)

    if "query" not in query_embeddings:
        logger.error("Failed to create embedding for the query. Exiting.")
        return 1

    query_embedding = query_embeddings["query"]
    logger.info("Successfully created embedding for the query.")

    # Step 4: Retrieve relevant documents using threading
    logger.info("Step 4: Retrieving relevant documents using threading...")
    document_retriever = DocumentRetriever(chunk_embeddings, chunks)
    relevant_chunks = document_retriever.retrieve_documents(
        query_embedding, top_k=args.top_k
    )

    if not relevant_chunks:
        logger.error("No relevant chunks found. Exiting.")
        return 1

    logger.info(f"Successfully retrieved {len(relevant_chunks)} relevant chunks.")

    # Step 5: Process the retrieved chunks asynchronously
    logger.info("Step 5: Processing the retrieved chunks asynchronously...")
    text_processor = TextProcessor()
    processed_chunks = await text_processor.process_chunks(relevant_chunks)

    if not processed_chunks:
        logger.error("Failed to process chunks. Exiting.")
        return 1

    logger.info(f"Successfully processed {len(processed_chunks)} chunks.")

    # Print the results to console
    print("\n" + "=" * 80)
    print(f"RESULTS FOR QUERY: '{args.query}'")
    print("=" * 80)

    for i, chunk in enumerate(processed_chunks):
        print(f"\nRESULT {i+1} (Similarity Score: {chunk['similarity']:.4f}):")
        print("-" * 80)
        print(
            f"Original Text: {chunk['original_text'][:500]}..."
            if len(chunk["original_text"]) > 500
            else f"Original Text: {chunk['original_text']}"
        )
        print("-" * 80)

    # Save results to output file
    output_path = os.path.join(
        "logs", f'query_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    )
    with open(output_path, "w") as f:
        f.write(f"RESULTS FOR QUERY: '{args.query}'\n")
        f.write("=" * 80 + "\n\n")

        for i, chunk in enumerate(processed_chunks):
            f.write(f"RESULT {i+1} (Similarity Score: {chunk['similarity']:.4f}):\n")
            f.write("-" * 80 + "\n")
            f.write(f"Original Text: {chunk['original_text']}\n\n")
            f.write(f"Processed Text: {chunk['processed_text']}\n")
            f.write("-" * 80 + "\n\n")

    logger.info(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
