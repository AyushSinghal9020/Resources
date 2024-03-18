import pandas as pd

import torch
from transformers import (
    AutoTokenizer ,
    AutoModelForCausalLM ,
    BitsAndBytesConfig
)

from peft import (
    prepare_model_for_kbit_training ,
    get_peft_model ,
    LoraConfig
)

import transformers

from datasets import Dataset , DatasetDict

from tqdm.notebook import tqdm


data = pd.read_csv('/content/sample.csv')
train_dataset_dict = Dataset.from_pandas(data)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
tokenizer.pad_token = tokenizer.eos_token

data = train_dataset_dict.map(lambda samples : tokenizer(samples['quote']) , batched = True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True ,
    bnb_4bit_use_double_quant = True ,
    bnb_4bit_quant_type = 'nf4' ,
    bnb_4bit_compute_dtype = torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/gpt-neox-20b' ,
    quantization_config = bnb_config ,
    device_map = {'' : 0}
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 8 ,
    lora_alpha = 32 ,
    target_modules = ['query_key_value'] ,
    lora_dropout = 0.05 ,
    bias = 'none',
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model , config)

args = transformers.TrainingArguments(
    per_device_train_batch_size = 1 ,
    gradient_accumulation_steps = 4 ,
    warmup_steps = 2 ,
    max_steps = 10 ,
    learning_rate = 2e-4 ,
    fp16 = True ,
    logging_steps = 1,
    output_dir = 'outputs' ,
    optim = 'paged_adamw_8bit'
)

collator = transformers.DataCollatorForLanguageModeling(
    tokenizer ,
    mlm = False
)


trainer = transformers.Trainer(
    model = model ,
    train_dataset = data ,
    args = args ,
    data_collator = collator ,
)

model.config.use_cache = False
trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model 
model_to_save.save_pretrained('outputs')

lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

text = 'What is Python Language'
device = 'cuda:0'

inputs = tokenizer(text , return_tensors = 'pt').to(device)
outputs = model.generate(**inputs , max_new_tokens = 20)