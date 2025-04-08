from src.utils.common import extract_data_from_pdf, split_text_into_chunks, download_embedding_model
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def main(data_path):
    """
    Extracts data from PDF files, splits the text into chunks, downloads an embedding model,
    and stores the embeddings in the Pinecone index.

    Args:
        data_path (str): The path to the directory containing the PDF files.

    Returns:
        None
    """

    logging.info("Extracting data from the pdf files...")
    # Extract data from the pdf files
    extracted_data = extract_data_from_pdf(data_path)

    logging.info("Splitting the text into chunks...")
    text_chunks = split_text_into_chunks(extracted_data)
    logging.info(f"Number of text chunks: {len(text_chunks)}")

    logging.info("Downloading the embedding model...")
    embeddings = download_embedding_model()

    logging.info("Storing the embeddings in the Pinecone index...")

    index_name=PINECONE_INDEX_NAME
    #Creating Embeddings for Each of The Text Chunks & storing
    vectorStore = PineconeVectorStore.from_texts([chunk.page_content for chunk in text_chunks], embeddings, index_name=PINECONE_INDEX_NAME)

    logging.info("Embeddings stored successfully in the Pinecone index.")

if __name__ == "__main__":
    data_path = "data/"
    main(data_path)