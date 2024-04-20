# -*- coding: utf-8 -*-
"""Authored by ShrugalTayal

This script performs information retrieval using the BM25 algorithm on a collection of documents stored in CSV files.

Dependencies:
    - rank_bm25: BM25 implementation for Python
    - json: JSON serialization and deserialization
    - re: Regular expression operations
    - nltk.corpus.stopwords: Collection of stopwords
    - nltk.tokenize.word_tokenize: Tokenization of words
    - nltk.stem.WordNetLemmatizer: Lemmatization of words
    - csv: CSV file reading and writing
    - os: Operating system interface

Functions:
    - read_csv(file_path): Read data from a CSV file and return a list of dictionaries representing documents.
    - preprocess_documents(documents): Preprocess document data extracted from CSV files by extracting relevant fields, such as titles, bodies, and page numbers.
    - preprocess_query(query): Preprocess the user query by converting it to lowercase, removing special characters and punctuation, tokenizing, removing stopwords, and lemmatizing the tokens.
    - tokenize_corpus(corpus): Tokenize the document corpus.
    - tokenize_query(query): Tokenize the user query.
    - initialize_bm25(tokenized_corpus): Initialize a BM25 model with the tokenized corpus.
    - get_document_scores(bm25_model, tokenized_query): Get BM25 scores for documents in the corpus based on the tokenized query.
    - retrieve_top_n_results(bm25_model, tokenized_query, corpus, n=1): Retrieve the top N documents from the corpus based on BM25 scores.
    - retrieve_top_n(corpus, query, n=1): Retrieve the top N documents from the corpus based on the user query.

Main:
    - The main section of the script reads data from CSV files located in a specified directory, preprocesses the documents and user query, performs information retrieval using BM25, and prints the retrieval results.

Usage:
    - Run the script, optionally providing a file path as input for the user query. If no input is given, a default query is used.
"""

from rank_bm25 import BM25Okapi
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv
import os
import spacy
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Read data from CSV file
def read_csv(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            documents.append(row)
    return documents
def generate_ngrams(text, n=2):
    """Generate unigrams and bigrams."""
    unigrams = text.split()
    bigrams = [' '.join(bg) for bg in ngrams(text.split(), 2)]
    return set(unigrams + bigrams)

def preprocess_documents(documents):
    # Extract relevant fields from documents
    titles = [doc['Title'] for doc in documents]
    bodies = [doc['Paragraph Text'] for doc in documents]
    pages = [doc['Page No'] for doc in documents]
    # You can preprocess the fields further if needed
    
    return titles, bodies, pages

def preprocess_query(query):
    # Convert query to lowercase
    query = query.lower()
    
    # Remove special characters and punctuation
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    
    # Tokenize the query
    tokens = word_tokenize(query)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join tokens back into a string
    preprocessed_query = ' '.join(lemmatized_tokens)
    
    return preprocessed_query

# def tokenize_corpus(corpus):
#     return [doc.split(" ") for doc in corpus]

# def tokenize_query(query):
#     return query.split(" ")

# def initialize_bm25(tokenized_corpus):
#     return BM25Okapi(tokenized_corpus)

# def get_document_scores(bm25_model, tokenized_query):
#     return bm25_model.get_scores(tokenized_query)

# def retrieve_top_n_results(bm25_model, tokenized_query, corpus, n=1):
#     return bm25_model.get_top_n(tokenized_query, corpus, n=n)

# def retrieve_top_n(corpus, query, n=1):
#     tokenized_corpus = tokenize_corpus(corpus)
#     tokenized_query = tokenize_query(query)
    
#     bm25 = initialize_bm25(tokenized_corpus)
#     doc_scores = get_document_scores(bm25, tokenized_query)
#     print('max doc_scores:', max(doc_scores))

#     if max(doc_scores) <= 0:
#         top_results = None
#     else:
#         top_results = retrieve_top_n_results(bm25, tokenized_query, corpus, n=n)
    
#     result_dict = {
#         "query": query,
#         "retrieval_results": top_results
#     }
    
#     return json.dumps(result_dict)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unnecessary components for speed

def preprocess_text(text):
    """Use SpaCy for preprocessing, disabling parser and NER for speed."""
    return [token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct]

def preprocess_documents_parallel(documents):
    """Parallel preprocessing of documents to improve speed."""
    with Pool(processes=os.cpu_count()) as pool:
        # Preprocess document bodies in parallel
        bodies_preprocessed = pool.map(preprocess_text, [doc['Paragraph Text'] for doc in documents])
    return bodies_preprocessed

def initialize_bm25_parallel(tokenized_corpus):
    """Initialize BM25 model, potentially in a parallelizable way for very large corpora."""
    return BM25Okapi(tokenized_corpus)

def retrieve_top_n_results_fast(bm25_model, tokenized_query, corpus, n=1):
    """Retrieve top N results more efficiently."""
    scores = bm25_model.get_scores(tokenized_query)
    top_n_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [corpus[idx] for idx in top_n_idxs]
def preprocess_text_spacy(text):
    """Preprocess text using SpaCy."""
    return ' '.join([token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct])
def filter_documents_by_query(inverted_index, query_tokens):
    """Filter documents that contain any of the query tokens."""
    doc_indices = set()
    for token in query_tokens:
        if token in inverted_index:
            doc_indices.update(inverted_index[token])
    return list(doc_indices)

# Adjust the main retrieval function to use the optimized components
def retrieve_top_n_optimized(corpus, query, n=1):
    # Preprocess query
    preprocessed_query = preprocess_text(query)
    
    # Tokenize corpus (assumed to be preprocessed)
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # Initialize BM25 model
    bm25 = initialize_bm25_parallel(tokenized_corpus)
    
    # Retrieve top n results using the optimized function
    top_results = retrieve_top_n_results_fast(bm25, preprocessed_query, corpus, n=n)
    
    return {"query": query,
        "retrieval_results": top_results}

def get_query_results(query,file_paths,top_k):
    dataset = []
    dataset_maplist = []
    #dir_path = os.path.join(os.path.dirname(os.getcwd()),'res','csv_etl_files')
    #print(dir_path)
    #dir_path = r'.\res\csv_etl_files'
    # List files present at the specified directory
    #csv_files = os.listdir(dir_path)
    # Join file path and file name for each file
    #file_paths = [os.path.join(dir_path, file) for file in csv_files]

    # Print the list of files
    #print("Files present", dir_path, ":")
    for file in file_paths:
        # print(file)
        documents = read_csv(file)
        dataset.extend(documents)
        dataset_maplist.extend([file] * len(documents))

    titles, bodies, pages = preprocess_documents(dataset)
    corpus = list(bodies)

    # If no input is given, assign a default value
    if not query:
        query = "Extension and amendment of Integrated Goods and Services Tax Act"

    preprocessed_query = preprocess_query(query)

        
    retrieval_results = retrieve_top_n_optimized(corpus, preprocessed_query,top_k)
    return retrieval_results
    