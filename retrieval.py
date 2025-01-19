from pypdf import PdfReader
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
import chromadb
from typing import List

def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a list of strings,
    where each string corresponds to the text of a single page.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - List[str]: A list containing the text content of each page in the PDF.
    """
    reader = PdfReader(file_path)

    # Extract text from each page and store it in a list
    text_by_page = [page.extract_text() for page in reader.pages]

    return text_by_page

def split_text_by_page(text_pages: List[str]):
    """
    Splits the text of each page into a list of non-empty substrings based on paragraphs ("\n\n").

    Parameters:
    - text_pages (List[str]): A list where each element is the text of a single page.

    Returns:
    - List[List[str]]: A nested list where each sublist contains non-empty substrings (paragraphs) from a single page.
    """
    return [[para for para in re.split('\n\n', page_text) if para.strip()] for page_text in text_pages]

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (Documents): A collection of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

def create_chroma_db(documents: List, path: str, name: str):
    """
    Creates a Chroma database using the provided documents, path, and collection name.

    Parameters:
    - documents: An iterable of documents to be added to the Chroma database.
    - path (str): The path where the Chroma database will be stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - Tuple[chromadb.Collection, str]: A tuple containing the created Chroma Collection and its name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, page in enumerate(documents):
        for j, paragraph in enumerate(page):
            document_id = f"page_{i}_para_{j}"
            db.add(documents=paragraph, ids=document_id)

    return db, name

def load_chroma_collection(path, name):
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

def get_relevant_passage(query, db, n_results):
    """
    Retrieves the most relevant passage(s) for a query from the Chroma database.

    Parameters:
    - query (str): The query text.
    - db (chromadb.Collection): The Chroma database collection.
    - n_results (int): Number of top results to retrieve.

    Returns:
    - str: The most relevant passage.
    """
    passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
    return passage
