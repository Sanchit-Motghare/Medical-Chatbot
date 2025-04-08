import os
from cv2 import log
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
import logging

from src.utils.common import download_embedding_model, set_env_variables
from src.prompts.question_maker import question_maker_prompt
from src.prompts.qna import qna_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Set the environment variables
logging.info("Setting environment variables...")
set_env_variables()


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Download the embedding model
logging.info("Downloading the embedding model...")
embeddings = download_embedding_model()
logging.info("Embedding model downloaded successfully.")

# Initialize the Pinecone vector store and retriever
logging.info("Initializing the Pinecone vector store and retriever...")
vectorStore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
retriever = vectorStore.as_retriever(search_kwargs={"k":3})
logging.info("Pinecone vector store and retriever initialized successfully.")

# Initialize the language model
logging.info("Initializing the language model...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
logging.info("Language model initialized successfully.")

# Initialize the output parser to convert the AI response to a string
output_parser = StrOutputParser()

# Define the question chain
logging.info("Initializing the question chain...")
question_chain = question_maker_prompt | llm | StrOutputParser()
logging.info("Question chain initialized successfully.")

def contextualized_question(inp: dict):
    """
    Determines the type of question to ask based on the input. If the input contains chat history, the question chain is returned. Otherwise, the question to ask is returned. Basically, it decides whether to reframe the question or not based on the chat history.

    Args:
        inp (dict): The input dictionary containing the chat history and question.

    Returns:
        str or langchain_core.runnables.Runnable: The question to ask or the question chain.

    """
    if inp.get("chat_history"):
        return question_chain
    else:
        return inp["question"]
    
# Define the retriever chain to retrieve the relevant documents from the Pinecone index
logging.info("Initializing the retriever chain...")
retriever_chain = RunnablePassthrough.assign(
        context=contextualized_question | retriever #| format_docs
    )
logging.info("Retriever chain initialized successfully.")

# Define the RAG chain
logging.info("Initializing the RAG chain...")
rag_chain = (
    retriever_chain
    | qna_prompt
    | llm
)
logging.info("RAG chain initialized successfully.")

def predict(message, history):
    """
    Predicts the AI response given a message and chat history.

    Args:
        message (str): The user's message.
        history (list): The chat history as a list of tuples containing the user's message and AI's response.

    Returns:
        str: The AI's response.

    """
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    ai_response = rag_chain.invoke({"question": message, "chat_history": history_langchain_format})
    return ai_response.content