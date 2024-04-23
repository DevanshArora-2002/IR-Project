import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK resources are available for tokenization
nltk.download('punkt')

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(tokens)

def retrieve_top_files(query, main_path,inverted_index, top_k=10):
    query_tokens = word_tokenize(preprocess_text(query))
    relevant_files = set()

    # Find files that contain at least one of the query terms using the inverted index
    for token in query_tokens:
        relevant_files.update(inverted_index.get(token, []))

    # If no relevant files, return empty list
    if not relevant_files:
        return []

    file_texts = []
    file_names = []
    relevant_files = [main_path+'/'+file for file in relevant_files]
    # Load and preprocess text from relevant files
    for file in relevant_files:
        df = pd.read_csv(file)
        file_text = " ".join(df["Paragraph Text"].astype(str).tolist())
        file_texts.append(preprocess_text(file_text))
        file_names.append(file)

    # Create TF-IDF vectors only for relevant files
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(file_texts)

    # Process the user query and transform it to the same TF-IDF space
    preprocessed_query = preprocess_text(query)
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity scores
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Pair file names with their respective similarity scores
    results = list(zip(file_names, similarity_scores[0]))

    # Sort by similarity scores in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Retrieve the top-k most similar files
    top_files = [file for file, score in sorted_results[:top_k]]

    return top_files

# Example usage (given that 'inverted_index' and 'csv_files' are defined):
# top_files = retrieve_top_files("example query", csv_files, inverted_index, 5)
