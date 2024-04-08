import re
import unicodedata
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import words
nltk.download('words')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import re
import unicodedata
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
word_list = set(words.words())
def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)

def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_acronyms(text):
    acronyms = {
        "asap": "as soon as possible",
        "fyi": "for your information",
        "eta": "estimated time of arrival",
        "rem":"return merchandise authorization",
        "sku":"stock keeping unit",
        "imho":"in my humble opinion"
    }
    for acronym, expansion in acronyms.items():
        text = re.sub(r"\b" + re.escape(acronym) + r"\b", expansion, text, flags=re.IGNORECASE)
    return text

def remove_special_chars(text):
    return re.sub('[^a-zA-Z0-9\s]', '', text)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)

def normalize_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_accented_chars(text)
    text = expand_acronyms(text)
    text = remove_special_chars(text)
    text = lemmatize_text(text)
    return text


from collections import Counter
import nltk
from nltk.corpus import words, wordnet
from nltk.stem import WordNetLemmatizer
import re

# Ensure the 'words' corpus is downloaded
nltk.download('words')
nltk.download('wordnet')  # Download the WordNet corpus for lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords list and the Punkt tokenizer models
nltk.download('stopwords')
nltk.download('punkt')


def remove_extra_spaces(paragraph):
    """
    Removes extra spaces from the given paragraph.

    Parameters:
    - paragraph (str): The paragraph from which to remove extra spaces.

    Returns:
    - str: The paragraph with extra spaces removed.
    """
    # Replace multiple spaces with a single space
    paragraph = re.sub(r'\s+', ' ', paragraph)

    # Remove leading and trailing spaces
    paragraph = paragraph.strip()

    return paragraph


def split_and_correct(word):
    if word in word_list:
        return word
    """Systematically attempt to split and correct a word by dictionary lookup."""
    # Adjusted base case to handle single letters correctly
    # Similar handling for punctuation as before
    if re.findall(r"[.;'(),]", word):
        # Split the word at any of the specified punctuation characters, keeping the characters in the result
        parts = re.split(r"([.;'(),])", word)

        # Define the characters to remove from the split results
        remove_chars = ".;()',"

        # Filter the split results to remove the specified punctuation characters and empty strings
        filtered_parts = [part for part in parts if part not in remove_chars and part != '']
        lst = []
        for part in filtered_parts:
            lst.append(split_and_correct(part))
        return ' '.join(lst)
    sol = None
    temp = ""
    for i in range(len(word)):
        piece1 = word[:i]
        piece2 = word[i:]
        if len(piece1) == 1 and piece1 == 'a':
            split = split_and_correct(piece2)
            if len(split) != 0:
                temp = piece1 + " " + split
                if sol is None:
                    sol = temp
                else:
                    sol = temp if len(temp) < len(sol) else sol

        if len(piece1) == 1 or len(piece2) == 1:
            continue

        if piece1 in word_list and piece2 in word_list:
            temp = piece1 + ' ' + piece2
            if sol is None:
                sol = temp
            else:
                sol = temp if len(temp) < len(sol) else sol

        elif piece1 in word_list:
            split = split_and_correct(piece2)
            if len(split) != 0:
                temp = piece1 + " " + split
                if sol is None:
                    sol = temp
                else:
                    sol = temp if len(temp) < len(sol) else sol

        elif piece2 in word_list:
            split = split_and_correct(piece1)
            if len(split) != 0:
                temp = split + " " + piece2
                if sol is None:
                    sol = temp
                else:
                    sol = temp if len(temp) < len(sol) else sol

    if sol is None:
        return ""
    return sol


def perform_correction(sentence):
    sentence = normalize_text(sentence)
    lst_words = list(sentence.split(" "))
    lst_corrected_words = []
    for word in lst_words:
        if word not in word_list:
            corrected_word = split_and_correct(word)
            lst = list(corrected_word.split(" "))
            lst_corrected_words.extend(lst)
        else:
            lst_corrected_words.append(word)

    str_ret = ""
    for i in range(len(lst_corrected_words)):
        str_ret = str_ret + lst_corrected_words[i] + " "
    return remove_extra_spaces(str_ret)