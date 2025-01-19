from pypdf import PdfReader
import re
from typing import List
from itertools import chain
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
import chromadb


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

def split_text(text_pages: List[str]) -> List[str]:
    """
    Splits the text of each page into a list of non-empty substrings based on paragraphs ("\n\n").
    Returns a flat list of strings (one for each paragraph across all pages).

    Parameters:
    - text_pages (List[str]): A list where each element is the text of a single page.

    Returns:
    - List[str]: A flat list containing non-empty substrings (paragraphs) from all pages.
    """
    # Flatten the list of lists into a single list
    return list(chain.from_iterable(
        [para for para in re.split('\n\n', page_text) if para.strip()] 
        for page_text in text_pages
    ))
    
def create_chroma_db(documents:List, path:str, name:str):
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

    for i, d in enumerate(documents):
        print(i)
        db.add(documents=d, ids=[str(i)])

    return db, name
    
    

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
        embd    = genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]
        return embd    

def make_database_for_rag(file_path,database_path,database_name):
    pdf_pages=load_pdf(file_path=file_path)
    create_chroma_db(pdf_pages,database_path,database_name)
    
    
    
    
      


    