import os 
import uuid
import PIL

from tqdm.notebook import tqdm

import google.generativeai as genai
from unstructured.partition.pdf import partition_pdf

from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_elements(pdf_paths) : 
    '''
    This function takes in a list of pdf paths and returns a list of pdf elements.

    Args:
    pdf_paths : list : List of paths to the pdfs.

    Returns:
    list : List of pdf elements.
    '''

    return [
        partition_pdf(
            filename = pdf_path ,
            extract_images_in_pdf = True ,
            infer_table_structure = True ,
            chunking_strategy = 'by_title' ,
            max_characters = 4000 ,
            new_after_n_chars = 3800 ,
            combine_text_under_n_chars = 2000 ,
            extract_block_output_dir = './images'
        )
        for pdf_path 
        in pdf_paths
    ]

def get_text_summaries(pdf_elements) : 
    '''
    This function takes in a list of pdf elements and returns a list of text summaries.

    Args:
    pdf_elements : list : List of pdf elements.

    Returns:
    list : List of text summaries.
    '''

    genai.configure(api_key = 'AIzaSyDyaY8u-BdM_IxU_YbJ5n4JLHy6A4uvxOE')
    model = genai.GeneriveModel('gemini-pro')

    return [
        model.generate_content(f'''
        Summarize this {pdf_element.text}
        ''').text
        for pdf_element 
        in pdf_elements
    ]

def get_image_summaries(pdf_elements) : 
    '''
    This function takes in a list of pdf elements and returns a list of image summaries.

    Args:
    pdf_elements : list : List of pdf elements.

    Returns:
    list : List of image summaries.
    '''

    genai.configure(api_key = 'AIzaSyDyaY8u-BdM_IxU_YbJ5n4JLHy6A4uvxOE')
    model = genai.GeneriveModel('gemini-pro-vision')

    image_paths = sorted(os.listdir('figures'))
    
    return [
        model.generate_content(
            PIL.Image.open(f'./images/{image_path}')
        ).text
        for image_path
        in image_paths
    ]

def get_documents(raw_pdf_elements , text_summaries , image_summaries) : 
    '''
    This function takes in a list of raw pdf elements, text summaries and image summaries and returns a list of documents.

    Args:
    raw_pdf_elements : list : List of raw pdf elements.
    text_summaries : list : List of text summaries.
    image_summaries : list : List of image summaries.

    Returns:
    list : List of documents.
    '''

    documents = []

    for element , summary in tqdm(
        zip(raw_pdf_elements , text_summaries) ,
        total = len(raw_pdf_elements) ,
    ) : 

        docouments.append(
            Document(
                page_content = summary , 
                metadata = {
                    'id' : str(uuid.uuid4()) , 
                    'type' : 'text' , 
                    'original_content' : element
                }
            )
        )

    for element , summary in tqdm(
        zip(raw_pdf_elements , image_summaries) ,
        total = len(raw_pdf_elements) ,
    ) : 

        documents.append(
            Document(
                page_content = summary , 
                metadata = {
                    'id' : str(uuid.uuid4()) , 
                    'type' : 'image' , 
                    'original_content' : PIL.Image.open(f'./images/{element}')
                }
            )
        )
  
    return documents

def get_vectorstore(documents) : 
    '''
    This function takes in a list of documents and returns a vectorstore.

    Args:
    documents : list : List of documents.

    Returns:
    vectorstore : Vectorstore : Vectorstore.
    '''

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(
        documents = documents ,
        embedding = embeddings
    )

    return vectorstore

def get_answer(vectorstore , question) : 
    '''
    This function takes in a vectorstore and a question and returns the answer.

    Args:
    vectorstore : Vectorstore : Vectorstore.
    question : str : Question.

    Returns:
    str : Answer.
    '''

    qa_chain = LLMChain(
        llm = ChatOpenAI(
            model = 'gpt-3.5-turbo' , 
            openai_api_key = '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' , 
            max_tokens = 1024
        ) , 
        prompt = PromptTemplate(
            '''
You are a vet doctor and an expert in analyzing dog's health.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible.
Answer:
            '''
        )
    )

    relevale_docs = vectorstore.similarity_search(question)
    context = ''
    relevant_images = []

    for doc in relevale_docs : 
        if doc.metadata['type'] == 'text' : context += '[text]' + str(doc.metadata['original_content'])
        else : 
            context += '[image]' + doc.page_content
            relevant_images.append(doc.metadata['original_content'])

    result = qa_chain.run({'context' : context, 'question' : question})

    return result , relevant_images