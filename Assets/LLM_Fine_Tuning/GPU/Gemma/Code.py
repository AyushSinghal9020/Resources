import torch
from transformers import (
    AutoTokenizer , 
    AutoModelForCausalLM , 
    BitsAndBytesConfig
)

from datasets import load_dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
import tqdm as notebook_tqdm
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from peft import LoraConfig, get_peft_model

import transformers

from trl import SFTTrainer

tokenizer = AutoTokenizer.from_pretrained('google/gemma' , add_eos_token = True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True , 
    bnb_4bit_use_double_quant = True ,
    bnb_4bit_quant_type = 'nf4' ,
    bnb_4bit_compute_dtype = torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    'google/gemma' , 
    quantization_config = bnb_config
    , device_map = {'' : 0}
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

cls = bnb.nn.Linear4bit
modules = set()

for name , module in model.named_modules() : 

    if isinstance(module , cls) : 

        names = name.split('.')
        modules.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in modules : modules.remove('lm_head')

lora_config = LoraConfig(
    r = 64 ,
    lora_alpha = 32 ,
    target_modules = modules ,
    lora_dropout = 0.05 ,
    bias = 'none' ,
    task_type = 'CAUSAL_LM'
)
model = get_peft_model(model, lora_config)

args = transformers.TrainingArguments(
    max_steps = 75 ,
    logging_steps = 1 ,
    warmup_steps = 0.03 ,
    learning_rate = 2e-4 ,
    output_dir = 'outputs' ,
    save_strategy = 'epoch' , 
    optim = 'paged_adamw_8bit' ,
    per_device_train_batch_size = 1 ,
    gradient_accumulation_steps = 4 
)

collator = transformers.DataCollatorForLanguageModeling(tokenizer , mlm = False)

trainer = SFTTrainer(
    args = args , 
    model = model , 
    data_collator = collator , 
    peft_config = lora_config , 
    train_dataset = hg_dataset , 
    dataset_text_field = 'text'  
)

trainer.train()

trainer.model.save_pretrained('gemma-Code-Instruct-Finetune-test')

lora_config_ = LoraConfig.from_pretrained(new_model)
model_tuned = get_peft_model(model, lora_config_)

def get_completion(query , model , tokenizer) :

    prompt_template = open('Prompt_Template.txt' , 'r').read().format(query = query)
    embeds = tokenizer(prompt_template , return_tensors = 'pt' , add_special_tokens = True)
    model_inputs = embeds.to('cuda:0')

    generated_ids = model.generate(
        **model_inputs ,
        do_sample = True ,
        max_new_tokens = 1000 ,
        pad_token_id = tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(generated_ids[0] , skip_special_tokens = True)
    
    return decoded

result = get_completion(
    query = open('Info.txt' , 'r').read() , 
    model = model_tuned , 
    tokenizer = tokenizer
)