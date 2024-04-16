from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import pandas as pd
import os
import csv

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_and_embed(texts):
    """
    Preprocess texts and generate BERT embeddings.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings
def read_csv(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            documents.append(row)
    return documents
def load_documents_from_csv():
    dir_path = os.path.join(os.path.dirname(os.getcwd()), 'res', 'csv_etl_files')
    csv_files = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, file) for file in csv_files]
    print("Files present", dir_path, ":")
    df_combined = None
    for file in file_paths:
        df = pd.read_csv(file)
        if df_combined is None:
            df_combined = df
        else:
            df_combined = pd.concat([df_combined, df], ignore_index=True)
    return df_combined

def query_and_retrieve(query,index, k=5):
    """
    Process a query, retrieve and rank documents based on semantic similarity.
    """
    query_embedding = preprocess_and_embed([query])
    distances, indices = index.search(query_embedding, k)
    return pd.DataFrame({
        'doc_index': indices.flatten(),
        'distance': distances.flatten()
    })

def get_query_results(query):
    df = load_documents_from_csv()
    documents = df['Paragraph Text'].tolist()
    doc_embeddings = preprocess_and_embed(documents)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    df_ret = query_and_retrieve(query,index)
    return df_ret
