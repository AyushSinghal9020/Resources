from tqdm.notebook import tqdm
import PIL
import os
import uuid
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

from helper import (
    get_pdf_elements , 
    get_text_summaries ,
    get_image_summaries ,
    get_documents ,
    get_vectorstore ,
    get_answer
)

def question_answer(raw_pdf_paths , question) : 
    '''
    This function takes in a list of raw pdf paths and a question and returns the answer.

    Args:
    raw_pdf_paths : list : List of raw pdf paths.
    question : str : Question.

    Returns:
    answer : str : Answer.
    '''

    raw_pdf_elements = get_pdf_elements(raw_pdf_paths)

    text_summaries = get_text_summaries(raw_pdf_elements)
    image_summaries = get_image_summaries(raw_pdf_elements)

    documents = get_documents(raw_pdf_elements , text_summaries , image_summaries)

    vectorstore = get_vectorstore(documents)

    answer = get_answer(vectorstore , question)

    return answer