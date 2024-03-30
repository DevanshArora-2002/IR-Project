from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
def get_top_retrieval(query,top_k=5):
    tokenizer = AutoTokenizer.from_pretrained("pile-of-law/legalbert-large-1.7M-2")
    model = AutoModel.from_pretrained("pile-of-law/legalbert-large-1.7M-2")

    csv_path = os.path.join(os.path.dirname(os.getcwd()), 'res', 'csv_etl_files')
    vector_path = os.path.join(os.path.dirname(os.getcwd()), 'res', 'Vector_Embedding')
    csv_data = os.listdir(csv_path)
    vector_data = os.listdir(vector_path)
    csv_df = None
    tensors = None
    for i in range(len(csv_data)):
        full_path = csv_path+"/"+csv_data[i]
        df = pd.read_csv(full_path)
        df = df['Paragraph Text']
        if csv_df is None:
            csv_df = df
        else:
            csv_df = pd.concat([csv_df,df],axis=0)

    for i in range(len(csv_data)):
        full_path = vector_path+"/"+csv_data[i][:-3]+"pt"
        tensor = torch.load(full_path)
        if tensors is None:
            tensors = tensor
        else:
            tensors = torch.concat((tensors,tensor),dim=0)


    query_inp = tokenizer(query,return_tensors='pt')
    with torch.no_grad():
        outputs = model(**query_inp)

    query_embedding = outputs.last_hidden_state[:,0,:]

    query_embedding_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    matrix_norm = torch.nn.functional.normalize(tensors, p=2, dim=1)

    # Compute cosine similarity
    cos_sim = torch.mm(matrix_norm, query_embedding_norm .transpose(0, 1))

    # Squeeze the result to remove extra dimensions and find top 5 indices
    top_k_values, top_k_indices = torch.topk(cos_sim.squeeze(), top_k)

    csv_dict = csv_df.iloc[top_k_indices].to_dict()
    return csv_dict




