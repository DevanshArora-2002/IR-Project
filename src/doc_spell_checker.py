from collections import Counter
import nltk
from nltk.corpus import words
import re

# Ensure the 'words' corpus is downloaded
nltk.download('words')

word_list = set(words.words())


def is_known(word):
    """Check if a word exists in the NLTK words dictionary."""
    return word.lower() in word_list and len(word) > 1  # Adjusted to avoid single-letter segments


def is_valid_single_letter(word):
    """Check if a single letter is a valid word ('a', 'I')."""
    return word.lower() in ['a', 'i']


def split_and_correct(word):
    """Systematically attempt to split and correct a word by dictionary lookup."""
    # Adjusted base case to handle single letters correctly
    if is_known(word) or len(word) <= 1 and is_valid_single_letter(word):
        return word

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

    best_correction = word  # Initialize with the original word as fallback
    best_length = 0  # Track the length of the best correction to prefer longer splits

    for i in range(1, len(word)):
        part1, part2 = word[:i], word[i:]

        corrected_part1 = split_and_correct(part1)
        corrected_part2 = split_and_correct(part2)

        if is_known(corrected_part1) and is_known(corrected_part2):
            return corrected_part1+' '+corrected_part2
        if is_known(corrected_part1):



    return best_correction.strip()  # Strip any leading/trailing spaces


# Example usage

test_words = ['word1word2', 'word1.word2', 'definitions.chapter', 'anexampletestword']

for word in test_words:
    corrected_word = split_and_correct(word)
    print(f"'{word}' corrected to: '{corrected_word}'")
