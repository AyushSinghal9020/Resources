import pandas as pd

import torch
from datasets import load_dataset , Dataset
from peft import (
    LoraConfig ,
    prepare_model_for_kbit_training ,
    get_peft_model
)

from transformers import (
    AutoTokenizer ,
    AutoModelForCausalLM ,
    GPTQConfig ,
    BitsAndBytesConfig , 
    TrainingArguments
)
import transformers

from trl import SFTTrainer

data = pd.read_csv('/content/data.csv')
data = Dataset.from_pandas(data)

tokenizer = AutoTokenizer.from_pretrained('TheBloke/zephyr-7B-alpha-GPTQ')
tokenizer.pad_token = tokenizer.eos_token

gpt_config = GPTQConfig(
    bits = 4 ,
    device_map = 'auto' ,
    use_cache = False ,
    lora_r = 16 ,
    lora_alpha = 16 ,
    tokenzier = tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    'TheBloke/zephyr-7B-alpha-GPTQ' ,
    quantization_config = gpt_config ,
    device_map = 'auto' ,
    use_cache = False ,
    trust_remote_code = True

model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r = 16 , 
    lora_alpha = 16 , 
    lora_dropout = 0.05 , 
    bias = 'none' , 
    task_type = 'CAUSAL_LM' , 
    target_modules = ['q_proj' , 'v_proj']
)

training_arguments = TrainingArguments(
    output_dir = 'zephyr-support-chatbot' , 
    per_device_train_batch_size = 8 , 
    gradient_accumulation_steps = 1 , 
    optim = 'paged_adamw_32bit' , 
    learning_rate = 2e-4 , 
    lr_scheduler_type = 'cosine' , 
    save_strategy = 'epoch' , 
    logging_steps = 50 , 
    num_train_epochs = 1 , 
    max_steps = 250 , 
    fp16 = True 
)

trainer = SFTTrainer(
    model = model , 
    train_dataset = processed_data , 
    peft_config = peft_config , 
    dataset_text_field = 'text' , 
    args = training_arguments , 
    tokenizer = tokenizer , 
    packing = False , 
    max_seq_length = 1024
)

trainer.train()