from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():

    query="What is service charge?"
    print('Received query: ', query)

    csv_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','csv_etl_files')
    csv_files = os.listdir(csv_path)
    csv_files = [csv_path+'/'+csv for csv in csv_files]
    inverted_index_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','inverted_index.pkl')
    print(inverted_index_path)
    with open(inverted_index_path, 'rb') as file:
        inverted_index = pickle.load(file)
    top_files = retrieve_top_files(query,csv_path,inverted_index,10)
    results = get_query_results(query,top_files,5)
    print('Results: ', results)
    print()

    prompt = prompt_generation(results['retrieval_results'], results['query'])

    print('Prompt: ', prompt)
    print()

    final_text = answer_generator(prompt,tokenizer, peft_model)
    final_text = print(final_text[len(prompt):].split('\n')[-1])
    print('Final answer: ', final_text)
    response = {"response": final_text}
    return response
    
    return {"message": 'hello world'}


@app.post("/chatbot")
async def predict(message: dict):
    query = message.get("message")
    # query="What is service charge?"
    print('Received query: ', query)

    csv_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','csv_etl_files')
    csv_files = os.listdir(csv_path)
    csv_files = [csv_path+'/'+csv for csv in csv_files]
    inverted_index_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','inverted_index.pkl')
    print(inverted_index_path)
    with open(inverted_index_path, 'rb') as file:
        inverted_index = pickle.load(file)
    top_files = retrieve_top_files(query,csv_path,inverted_index,10)
    results = get_query_results(query,top_files,5)
    print('Results: ', results)
    print()

    prompt = prompt_generation(results['retrieval_results'], results['query'])

    print('Prompt: ', prompt)
    print()

    final_text = answer_generator(prompt,tokenizer, peft_model)
    final_text = print(final_text[len(prompt):].split('\n')[-1])
    print('Final answer: ', final_text)
    response = {"response": final_text}
    
    return response



import pandas as pd
import pickle
import json
import sys
import os

import nltk
nltk.download('omw-1.4')

sys.path.append('./src')

from document_retrieval_BM25 import get_query_results
from doc_spell_checker import perform_correction
from tf_idf_retrieval import retrieve_top_files

from pprint import pprint
import torch
import torch.nn as nn
import transformers
from huggingface_hub import notebook_login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from peft import TaskType, PeftModel
from transformers import BitsAndBytesConfig


model_path = "mistralai/Mistral-7B-v0.1"

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map = 'auto',
    trust_remote_code=True,
    token='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'
)

repo_name = "DevanshArora2002/legal-PPO-model"
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# Load the PEFT model from the Hugging Face Hub
peft_model = PeftModel.from_pretrained(
    base_model,
    repo_name,
    #quantization_config=bnb_config,
    lora_config=lora_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, token='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')

def answer_generator(prompt,tokenizer,model):
    """
    Generate text from a pre-trained language model given a prompt and a model name.

    Parameters:
    prompt (str): The prompt text to feed to the language model.
    model_name (str): The model identifier on Hugging Face's model hub.

    Returns:
    str: The text generated by the model.
    """
    print(model.device)
    
    input = tokenizer(prompt,return_tensors='pt')
    input_ids = input['input_ids'].to(model.device)
    attention_mask = input['attention_mask'].to(model.device)
    out = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens=200,
        temperature=0.5,
        no_repeat_ngram_size=2,
        top_p=0.9,
        do_sample=True,
    )
    output_text = tokenizer.decode(out[0][-200:],skip_special_tokens=True)
    return output_text

def prompt_generation(ranked_text,query):
    context = ""
    for i in range(len(ranked_text)):
        context += f"{i+1}: {ranked_text[i]}"
    prompt = f"""Generate legal advice for {query} using the following contexual information {context}"""
    return prompt


print('Outside Backend')
query="What is service charge?"
print('Received query: ', query)

csv_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','csv_etl_files')
csv_files = os.listdir(csv_path)
csv_files = [csv_path+'/'+csv for csv in csv_files]
inverted_index_path = os.path.join(os.path.dirname(os.getcwd()),'backend', 'res','inverted_index.pkl')
print(inverted_index_path)
with open(inverted_index_path, 'rb') as file:
    inverted_index = pickle.load(file)
top_files = retrieve_top_files(query,csv_path,inverted_index,10)
results = get_query_results(query,top_files,5)
print('Results: ', results)
print()

prompt = prompt_generation(results['retrieval_results'], results['query'])

print('Prompt: ', prompt)
print()

final_text = answer_generator(prompt,tokenizer, peft_model)
final_text = print(final_text[len(prompt):].split('\n')[-1])
print('Final answer: ', final_text)