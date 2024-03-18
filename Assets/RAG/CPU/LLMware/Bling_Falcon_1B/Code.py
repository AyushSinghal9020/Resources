print('Importing libraries...')

import os
from tqdm import tqdm
from PyPDF2 import PdfReader

from llmware.prompts import Prompt

def file_is_of_correct_format(file_name) : 
    '''
    This function checks if the file is of the correct format

    Args :
    file_name : str : The name of the file

    Returns :
    bool : True if the file is of the correct format, False otherwise
    '''

    return file_name.endswith('.txt') or file_name.endswith('.pdf')

def extract_text_from_txt(file_name) : 
    '''
    This function extracts the text from a .txt file

    Args :
    file_name : str : The name of the file

    Returns :
    str : The text from the file
    '''

    text = open(f'Files/{file_name}', 'r').read()

    return text 

def extract_text_from_pdf(file_name) : 
    '''
    This function extracts the text from a .pdf file

    Args :
    file_name : str : The name of the file

    Returns :
    str : The text from the file
    '''

    reader = PdfReader(f'Files/{file_name}')

    text = ''

    for page in reader.pages : text += page.extract_text()

    return text

def extract_text_from_files() : 
    '''
    This function extracts the text from all the files in the Files directory

    Returns :
    str : The text from the files
    '''

    files = os.listdir('Files')

    valid_files = [
        file_ 
        for file_ 
        in files 
        if file_is_of_correct_format(file_)
    ]

    text = '' 

    for files in valid_files : 

        if files.endswith('.txt') : text += extract_text_from_txt(files)
        elif files.endswith('.pdf') : text += extract_text_from_pdf(files)
        elif files.endswith('.doc') or files.endswith('.docx') : text += extract_text_from_doc(files)
        elif files.endswith('.ppt') or files.endswith('.pptx') : text += extract_text_from_ppt(files)
        
    return text

def ask_question(questions) : 
    '''
    This function asks a question to the model

    Args :
    questions : list : The list of questions to ask

    Returns :
    list : The list of answers to the questions
    '''

    text = extract_text_from_files()

    print('Loading Model LLMware Bling 1B 0.1')
    print('Download will only happen once, it can take time depending on the Internet Speed (4.1 GB)')

    prompt = Prompt()
    prompt = prompt.load_model('llmware/bling-falcon-1b-0.1')

    print('Model Loaded')

    answers = []

    for question in tqdm(questions , total = len(questions)) : 

        text_chunks = [
            text[index : index + 2047]
            for index
            in range(0, len(text), 2047)
        ]

        for text in text_chunks : 

            answer = prompt.prompt_main(prompt = question, context = text)

            if answer['llm_response'] != 'Not Found' : 

                answers.append(answer['llm_response'])
                break

    return answers
    
questions = []
print("Enter ''(nothing) to stop")

while True : 

    question = input('Enter the question : ')
    questions.append(question)

    if question == '' : break

answers = ask_question(questions)

for question , answer in zip(questions, answers) : print(f'{question} : {answer}')