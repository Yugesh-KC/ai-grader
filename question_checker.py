from database_maker import GeminiEmbeddingFunction
import chromadb
from itertools import chain
import os
import google.generativeai as genai


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


def get_relevant_passage(query, db, n_results=3):
  passage = db.query(query_texts=query, n_results=n_results)['documents']
  passage = list(chain.from_iterable(passage))

  return passage

def make_rag_prompt(query, full_marks, ideal_answer, relevant_text, students_answer):
    """
    Generates a grading prompt for a teacher checking an engineering exam paper.
    
    Parameters:
    - query: The exam question.
    - full_marks: The full marks allocated for the question.
    - ideal_answer: The ideal or model answer for the question.
    - relevant_text: The relevant reference text to be used in grading.
    - students_answer: The answer provided by the student.

    Returns:
    - A formatted prompt that can be used to grade the student's answer.
    """
    # Escape special characters to ensure proper formatting
    escaped_relevant_text = relevant_text.replace("'", "").replace('"', "").replace("\n", " ")
    if ideal_answer:
        ideal_answer = ideal_answer.replace("'", "").replace('"', "").replace("\n", " ")

    # Format the grading prompt with all the necessary information
    prompt = f"""
    You are a teacher checking Bachelor's in Engineering exam papers. You will be given a question, its full marks, the ideal answer, 
    the relevant reference text, and the answer given by the student. Your task is to grade the student's answer strictly, keeping in mind 
    the full marks allocated for the question. Do not use your own knowledge and logic just focus mainly on the ideal answer.
    to evaluate the answer. 

    Be sure to evaluate the completeness, accuracy, and clarity of the student's response while being fair and consistent with the marks.

    QUESTION: '{query}'
    Full Marks: {full_marks}
    Ideal Answer: '{ideal_answer}'
    Relevant Reference Text: '{escaped_relevant_text}'
    
    Student's Answer: '{students_answer}'

    GRADE:
    """

    return prompt

def generate_answers(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def check_answer(db_path,db_name, query, full_marks, students_answer, ideal_answer=None, rag_chunks=3):
    """
    Generates an answer based on a student's response, relevant reference text, and an ideal answer.
    
    Parameters:
    - db_path: THe path of the chroma db
    - db_name: The Chroma database for retrieving relevant text.
    - query: The exam question.
    - full_marks: The full marks allocated for the question.
    - students_answer: The answer provided by the student.
    - ideal_answer: The ideal answer, if available (defaults to None).
    - n_results: The number of relevant text chunks to retrieve (default is 3).
    
    Returns:
    - The generated grade or evaluation based on the prompt.
    """
    
    db=load_chroma_collection(db_path,db_name)
    # Retrieve the top N relevant text chunks for the query
    relevant_text_chunks = get_relevant_passage(query, db, n_results=rag_chunks)
    
    # If no relevant text is found, return a default message
    # if not relevant_text_chunks:
    #      "No relevant information found for grading."

    # Combine the retrieved text chunks into one passage
    relevant_text=""
    if relevant_text_chunks:
        relevant_text = " ".join(relevant_text_chunks)
    
    
    
    # Generate the grading prompt using the ideal answer (if available) and student answer
    prompt = make_rag_prompt(query, full_marks, ideal_answer=ideal_answer,  relevant_text=relevant_text, students_answer=students_answer)
    
    # Generate the answer or evaluation from the model based on the prompt
    answer = generate_answers(prompt)
    
    return answer


print(check_answer(db_path='try_database',db_name='try4',query='how many grams in a kilogram',ideal_answer="20000kg",full_marks=2,students_answer="20000g",rag_chunks=5))