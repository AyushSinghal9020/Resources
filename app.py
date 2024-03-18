import streamlit as st 
import os 

def write_code(path) : 

    files = os.listdir(path)

    for file in files :

        try : st.code(open(f'{path}/{file}').read())
        except : pass

def rag() : 

    device = st.selectbox('Select the Device' , ['CPU' , 'GPU'])

    if device == 'CPU' : 

        ram = st.selectbox('Select the RAM' , ['1 - 3' , '4 - 6'])

        if ram == '1 - 3' : 

            host = st.selectbox('Select the host' , ['LLMware'])

            if host == 'LLMware' : 

                model_map = {
                    'Bling 1B' : 'Assets/RAG/CPU/LLMware/Bling_1B' , 
                    'Bling 1.4B' : 'Assets/RAG/CPU/LLMware/Bling_1.4B' , 
                    'Bling Falcon 1B' : 'Assets/RAG/CPU/LLMware/Bling_Falcon_1B' , # Assets/RAG/CPU/LLMware/Bling_Falcon_1B
                    'Bling Sheared Llama 1.3B' : 'Assets/RAG/CPU/LLMware/Bling_Sheared_Llama_1.3B' , 
                    'Bling Sheared Llama 2.7B' : 'Assets/RAG/CPU/LLMware/Bling_Sheared_Llama_2.7B' ,  
                    'Bling Red Pajamas 3B' : 'Assets/RAG/CPU/LLMware/Bling_Red_Pajamas_3B' , 
                    'Bling Stable LM 3B 4eit' : 'Assets/RAG/CPU/LLMware/Bling_Stable_LM_3B_4eit'
                }

                model = st.selectbox('Select the model' , list(model_map.keys()))
                write_code(model_map[model])
            else : pass
        elif ram == '4 - 6' : 

            host = st.selectbox('Select the host' , ['GPT'])

            if host == 'GPT' :

                modal = st.selectbox('Select the modal' , ['MultiModal'])

                if modal == 'MultiModal' :
                    
                    model_map = {'3.5' : 'Assets/RAG/CPU/GPT/Multi_Modal/3_5' , }
                    model = st.selectbox('Select the model' , list(model_map.keys()))
                    write_code(model_map[model])
                else : pass
            else : pass
        else : pass
    else : pass

def llm_fine_tuning() : 

    device = st.selectbox('Select the Device' , ['CPU' , 'GPU'])

    if device == 'CPU' : pass
    else : 

        host = st.selectbox('Select the Host' , ['Gemma' , 'GPT' , 'Llama' , 'Phi' , 'Zypher'])

        if host == 'Gemma' : write_code('Assets/LLM_Fine_Tuning/GPU/Gemma')
        elif host == 'GPT' : 

            model_map = {'NEOX 20B' : 'Assets/LLM_Fine_Tuning/GPU/GPT/NEOX_20B'}
            model = st.selectbox('Select the Model' , list(model_map.keys()))
            write_code(model_map[model])
        elif host == 'Llama' : write_code('Assets/LLM_Fine_Tuning/GPU/Llama')
        elif host == 'Phi' : 

            model_map = {'2' : 'Assets/LLM_Fine_Tuning/GPU/Phi/2'}
            model = st.selectbox('Select the Model' , list(model_map.keys()))
            write_code(model_map[model])
        elif host == 'Zypher' : 

            model_map = {'7B Alpha GPTQ' : 'Assets/LLM_Fine_Tuning/GPU/Zypher/7B_Alpha_GPTQ' ,}
            model = st.selectbox('Select the Model' , list(model_map.keys()))
            write_code(model_map[model])
        else : pass


option = st.sidebar.selectbox(
    'Select the Option' , 
    [
        'RAG' , 
        'LLM Fine Tuning'
    ])

if option == 'RAG' : rag()
elif option == 'LLM Fine Tuning' : llm_fine_tuning()
