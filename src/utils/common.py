# This file will contain common functions that are used in multiple places in the codebase

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import yaml

# Function to extract data from the pdf file
def extract_data_from_pdf(data_dir):
    """
    Function to extract data from the pdf file
    
    Args:
    data_dir : str : Path to the directory containing the pdf files
    
    Returns:
    documents : list : List of documents extracted from the pdf files
    """
    
    loader = DirectoryLoader(data_dir, glob = "*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split the text into chunks
def split_text_into_chunks(documents):
    """
    Function to split the text into chunks
    
    Args:
    documents : list : List of documents
    
    Returns:
    chunks : list : List of chunks with text split into chunks
    """
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 20)
    chunks = splitter.split_documents(documents)
    return chunks

# Function to download the embedding model
def download_embedding_model():
    """
    Function to download the embedding model
    
    Args:
    model_name : str : Name of the model to download
    
    Returns:
    embeddings : object : Embedding model object
    """
    
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    return embeddings

def set_env_variables():
    """
    Sets environment variables based on the values specified in the config file.

    Reads the config file located at './config/config.yaml' and sets each key-value pair as an environment variable.

    Example:
        If the config file contains the following key-value pairs:
        ```
        DATABASE_HOST: 'localhost'
        DATABASE_PORT: '5432'
        ```
        The function will set the environment variables as follows:
        ```
        os.environ['DATABASE_HOST'] = 'localhost'
        os.environ['DATABASE_PORT'] = '5432'
        ```
    """
    path_config = "./config/config.yaml"
    # Load YAML file
    with open(path_config, 'r') as file:
        config = yaml.safe_load(file)

    # Set environment variables
    for key, value in config.items():
        os.environ[key] = value