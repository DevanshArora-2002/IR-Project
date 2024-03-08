# -*- coding: utf-8 -*-
"""Authored by ShrugalTayal
"""

from bs4 import BeautifulSoup
import cssutils
import fitz
import pandas as pd
import os
from nltk.util import ngrams
import nltk
from nltk.tokenize import word_tokenize
# Download NLTK tokenizer data
nltk.download('punkt')
from rank_bm25 import BM25Okapi

# Document Extraction Transformation and Loading
def adding_context_block(inp):
    context=[]
    if(len(inp)==0):
        return context
    inp[0]['context']=' '
    context.append(inp[0])
    for i in range(1,len(inp)):
        prev_font=float(inp[i-1]['font_size'][:-2])
        curr_font=float(inp[i]['font_size'][:-2])

        if(prev_font==curr_font):
            inp[i]['context']=inp[i-1]['context']
        elif(prev_font>=curr_font):
            inp[i]['context']=inp[i-1]['context']+', '+inp[i-1]['text']
        else:
            inp[i]['context']=''

        context.append(inp[i])
    return context

def grouping(inp):
    output = []
    current_font_size = None
    current_paragraph = ""

    for span in inp:
        if current_font_size is None:
            current_font_size = span[1]
            current_paragraph = span[0]
            current_color = span[2]
            current_focus_words = " "
        elif current_font_size == span[1] and span[2]==current_color:
            current_paragraph += " " + span[0]
        elif current_font_size == span[1]:
            current_paragraph += " "  + span[0]
            current_focus_words += ", " + span[0]
        else:
            output.append({"text": current_paragraph, "font_size": current_font_size,
                           'focus_words':current_focus_words})

            current_font_size = span[1]
            current_paragraph = span[0]
            current_focus_words=' '
            current_color= span[2]
    if current_paragraph:
        output.append({"text": current_paragraph, "font_size": current_font_size,
                      'focus_words':current_focus_words})

    return output

def get_html(page):
    html_text=page.get_text('html')
    soup = BeautifulSoup(html_text, 'html.parser')
    text_info=[]
    p_tags = soup.find_all('p')
    unique_spans = set()
    unique_lst=[]
    for p in p_tags:
        span_tags = p.find_all('span')
        for span in span_tags:
            text = span.text
            style = span.get('style', '')
            font_size = 16
            font_color = '#000000'

            if style:
                style_list = style.split(';')
                for s in style_list:
                    if 'font-size' in s:
                        font_size = s.split(':')[1].strip()
                    if 'color' in s:
                        font_color = s.split(':')[1].strip()

            if(font_color=='#ffffff'):
                continue
            # Create a unique identifier for each span
            unique_id = (text, font_size, font_color)

            # Only process this span if we haven't seen it before
            if unique_id not in unique_spans:
                #print(f'Span text: {text}, Span Font Size: {font_size}, Span Font Color: {font_color}')
                unique_spans.add(unique_id)
                unique_lst.append(unique_id)

    if(len(unique_lst)==0 or len(unique_lst)==1):
        return unique_lst
    if(unique_lst[-1][1]!=unique_lst[-2][1]):
        font_val_curr=float(unique_lst[-1][1][:-2])
        font_val_prev=float(unique_lst[-2][1][:-2])

        if(font_val_curr<font_val_prev):
            unique_lst=unique_lst[:-1]

    return unique_lst

def combine(book_path):
    doc=fitz.open(book_path)
    doc_info={'text':[],
             'font':[],
             'focus':[],
             'context':[],
             'page_no':[]}
    for page_no in range(1,len(doc)-1):
        print(page_no)
        page=doc[page_no]
        html_info=get_html(page)
        text_grouping=grouping(html_info)
        context_add=adding_context_block(text_grouping)

        for o in context_add:
            doc_info['text'].append(o['text'])
            doc_info['font'].append(o['font_size'])
            doc_info['focus'].append(o['focus_words'])
            doc_info['context'].append(o['context'])
            doc_info['page_no'].append(page_no)
    df=pd.DataFrame.from_dict(doc_info)
    return df

"""
# Specify the path to the content folder
content_folder = '/content'

# List all files in the content folder
files = os.listdir(content_folder)

# Filter out directories
files = [file for file in files if os.path.isfile(os.path.join(content_folder, file))]

# Initialize an empty DataFrame
df = pd.DataFrame()

for file in files:
  print('/content/{}'.format(file))
  book_path = '/content/{}'.format(file)
  df_temp = combine(book_path)
  #print(file)
  #print(df_temp)
  df = pd.concat([df, df_temp], ignore_index=True)

pd.set_option('display.max_rows', None)
df
"""

"""# N-gram Generation for Text Data"""
# Function to create n-grams
def create_ngrams(text, n):
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in n_grams]

# Function to create bigrams
def create_bigrams(df, text_column):
    df['bigrams'] = df[text_column].apply(lambda x: create_ngrams(x, 2))
    return df

# Function to create trigrams
def create_trigrams(df, text_column):
    df['trigrams'] = df[text_column].apply(lambda x: create_ngrams(x, 3))
    return df

# Sample DataFrame (if needed)
"""
data = {
    'text': ['2 CHAPTER V O', 'FFENCES AND PENALTIES', 'S', 'ECTIONS', '25. Penalty for use of non-standard weight or ...', '3', 'THE LEGAL METROLOGY ACT, 2009 A', 'CT', 'N'],
    'font': ['11.0pt', '9.0pt', '11.0pt', '9.0pt', '11.0pt', '11.0pt', '12.0pt', '10.0pt', '12.0pt'],
    'focus': ['', '', '', '', '', '', '', '', ''],
    'context': ['', ', 2 CHAPTER V O', '', ', S', '', '', ', THE LEGAL METROLOGY ACT, 2009 A', '', ''],
    'page_no': [1, 1, 1, 1, 1, 2, 2, 2, 2]
}
df = pd.DataFrame(data)
"""
"""
# Call the functions with appropriate parameters
df = create_bigrams(df, 'text')
df = create_trigrams(df, 'text')

# Display the DataFrame with n-grams
print(df[['text', 'bigrams', 'trigrams']])
"""

"""# BM25 Text Ranking"""
def preprocess_text(df, text_column):
    # Tokenize text
    df['tokenized_text'] = df[text_column].apply(lambda x: word_tokenize(x.lower()))
    return df

def calculate_bm25(df, tokenized_text_column, query):
    # Build BM25 index
    bm25 = BM25Okapi(df[tokenized_text_column].tolist())

    # Tokenize and lowercase query
    query_tokens = word_tokenize(query.lower())

    # Get BM25 scores for the query
    scores = bm25.get_scores(query_tokens)

    # Add BM25 scores to DataFrame
    df['bm25_score'] = scores

    # Sort DataFrame by BM25 scores
    df = df.sort_values(by='bm25_score', ascending=False)

    return df[['text', 'bm25_score']]
"""
# Sample DataFrame
data = {
    'text': ['2 CHAPTER V O', 'FFENCES AND PENALTIES', 'S', 'ECTIONS', '25. Penalty for use of non-standard weight or ...', '3', 'THE LEGAL METROLOGY ACT, 2009 A', 'CT', 'N'],
    'font': ['11.0pt', '9.0pt', '11.0pt', '9.0pt', '11.0pt', '11.0pt', '12.0pt', '10.0pt', '12.0pt'],
    'focus': ['', '', '', '', '', '', '', '', ''],
    'context': ['', ', 2 CHAPTER V O', '', ', S', '', '', ', THE LEGAL METROLOGY ACT, 2009 A', '', ''],
    'page_no': [1, 1, 1, 1, 1, 2, 2, 2, 2]
}
df = pd.DataFrame(data)
"""

"""
# Tokenize and preprocess text
df = preprocess_text(df, 'text')

# Calculate BM25 scores for the query
query = "penalty for use of non-standard weight"
result_df = calculate_bm25(df, 'tokenized_text', query)

print(result_df)
"""

"""# BM25 Text Ranking with Trigrams"""
def preprocess_data(data):
    """
    Preprocess the input data and create a pandas DataFrame.
    """
    df = pd.DataFrame(data)
    return df

def generate_ngrams(text, n):
    """
    Generate n-grams of variable length from the input text.
    """
    tokens = word_tokenize(text.lower())
    return [' '.join(gram) for gram in ngrams(tokens, n)]

def calculate_bm25_scores(df, query, n_values=[3]):
    """
    Calculate BM25 scores for each document in the DataFrame.
    """
    all_scores = []

    for n in n_values:
        df[f'ngrams_{n}'] = df['text'].apply(lambda x: generate_ngrams(x, n))
        query_tokens = word_tokenize(query.lower())
        doc_ngrams = df[f'ngrams_{n}'].tolist()
        bm25 = BM25Okapi(doc_ngrams)
        scores = bm25.get_scores(query_tokens)
        all_scores.extend(scores)

    df['bm25_score'] = all_scores
    df = df.sort_values(by='bm25_score', ascending=False)
    return df
"""
# Sample DataFrame
data = {
    'text': ['2 CHAPTER V O', 'FFENCES AND PENALTIES', 'S', 'ECTIONS', '25. Penalty for use of non-standard weight or ...', '3', 'THE LEGAL METROLOGY ACT, 2009 A', 'CT', 'N'],
    'font': ['11.0pt', '9.0pt', '11.0pt', '9.0pt', '11.0pt', '11.0pt', '12.0pt', '10.0pt', '12.0pt'],
    'focus': ['', '', '', '', '', '', '', '', ''],
    'context': ['', ', 2 CHAPTER V O', '', ', S', '', '', ', THE LEGAL METROLOGY ACT, 2009 A', '', ''],
    'page_no': [1, 1, 1, 1, 1, 2, 2, 2, 2]
}

# Preprocess data
df = preprocess_data(data)
"""
"""
# Define query
query = "penalty for use of non-standard weight"

# Calculate BM25 scores
df_with_scores = calculate_bm25_scores(df, query)

# Display the DataFrame with BM25 scores
print(df_with_scores[['text', 'bm25_score']])
"""