# =========================================================
# 📌 IMPORT LIBRARIES
# =========================================================

# pip install nltk scikit-learn
# Install required libraries:
# nltk -> Natural Language Processing
# scikit-learn -> TF-IDF vectorization

import nltk  # NLP library

from nltk.tokenize import word_tokenize

# Splits sentence into words/tokens

from nltk.corpus import stopwords

# Provides stopword list

from nltk.stem import PorterStemmer, WordNetLemmatizer

# PorterStemmer -> stemming
# WordNetLemmatizer -> lemmatization

from nltk import pos_tag

# Performs Part-of-Speech tagging

from sklearn.feature_extraction.text import TfidfVectorizer

# Used for TF-IDF representation


# =========================================================
# 📌 DOWNLOAD REQUIRED NLTK DATA (RUN ONLY FIRST TIME)
# =========================================================

nltk.download("punkt_tab")
# Downloads tokenizer table

nltk.download("punkt")
# Downloads tokenizer package

nltk.download("stopwords")
# Downloads stopword dataset

nltk.download("averaged_perceptron_tagger")
# Downloads POS tagging model

nltk.download("wordnet")
# Downloads WordNet dataset for lemmatization

nltk.download("averaged_perceptron_tagger_eng")
# Downloads English POS tagger


# =========================================================
# 📌 SAMPLE DOCUMENT
# =========================================================

text = """Data science is an interdisciplinary field that uses scientific methods, 
processes, algorithms and systems to extract knowledge and insights from structured 
and unstructured data."""
# Sample text/document for NLP processing

print("\nOriginal Text:\n", text)
# Output: Displays original text


# =========================================================
# 📌 1. TOKENIZATION
# =========================================================

tokens = word_tokenize(text)
# Splits text into words/tokens

print("\nTokens:\n", tokens)
# Output: Displays tokenized words


# =========================================================
# 📌 2. POS TAGGING
# =========================================================

pos_tags = pos_tag(tokens)
# Assigns grammatical tags to each word

print("\nPOS Tags:\n", pos_tags)
# Output: Displays words with POS tags

# Example:
# NN -> Noun
# VB -> Verb
# JJ -> Adjective


# =========================================================
# 📌 3. STOPWORDS REMOVAL
# =========================================================

stop_words = set(stopwords.words("english"))
# Loads English stopwords

filtered_words = [word for word in tokens if word.lower() not in stop_words]
# Removes common unnecessary words like:
# is, an, and, the etc.

print("\nAfter Stopword Removal:\n", filtered_words)
# Output: Displays filtered meaningful words


# =========================================================
# 📌 4. STEMMING
# =========================================================

stemmer = PorterStemmer()
# Creates stemming object

stemmed_words = [stemmer.stem(word) for word in filtered_words]
# Converts words into root/base form

print("\nStemmed Words:\n", stemmed_words)
# Output: Displays stemmed words

# Example:
# learning -> learn
# processes -> process


# =========================================================
# 📌 5. LEMMATIZATION
# =========================================================

lemmatizer = WordNetLemmatizer()
# Creates lemmatizer object

lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
# Converts words into meaningful dictionary root form

print("\nLemmatized Words:\n", lemmatized_words)
# Output: Displays lemmatized words


# =========================================================
# 📌 6. TF-IDF REPRESENTATION
# =========================================================

# Using multiple documents for better TF-IDF
documents = [
    text,
    "Data science involves machine learning and data analysis",
    "Machine learning is a part of data science",
]
# List of sample documents

vectorizer = TfidfVectorizer()
# Creates TF-IDF vectorizer object

tfidf_matrix = vectorizer.fit_transform(documents)
# Converts documents into TF-IDF matrix

# Feature names (words)
feature_names = vectorizer.get_feature_names_out()
# Extracts all unique words/features

print("\nTF-IDF Feature Names:\n", feature_names)
# Output: Displays all important words/features

print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())
# Output: Displays TF-IDF score matrix


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. Original Text Output
# Displays the original input text/document used for NLP processing.

# 2. Tokenization Output
# Splits sentence into individual words/tokens.

# Example:
# ['Data', 'science', 'is', 'an', ...]

# Tokenization is first preprocessing step in NLP.

# 3. POS Tagging Output
# Displays grammatical role of each word.

# Example:
# ('Data', 'NN')
# NN -> Noun
# VB -> Verb
# JJ -> Adjective

# Helps understand sentence structure.

# 4. Stopword Removal Output
# Removes unnecessary common words like:
# is, an, the, and etc.

# Keeps only meaningful words for analysis.

# 5. Stemming Output
# Converts words into root/base form.

# Example:
# learning -> learn
# algorithms -> algorithm

# Stemmed words may not always be proper dictionary words.

# 6. Lemmatization Output
# Converts words into meaningful dictionary root form.

# Example:
# studies -> study
# running -> run

# Lemmatization gives more meaningful results than stemming.

# 7. TF-IDF Feature Names Output
# Displays all unique important words from documents.

# Example:
# ['analysis', 'data', 'learning', 'machine', ...]

# These words are used as features in NLP models.

# 8. TF-IDF Matrix Output
# Displays numerical TF-IDF scores for each word in each document.

# Formula:
# TF-IDF = TF × IDF

# TF -> Term Frequency
# IDF -> Inverse Document Frequency

# High TF-IDF score means:
# Word is important in a document
# but not common in all documents.

# Example:
# If "machine" appears frequently in one document
# but rarely in others,
# it gets higher TF-IDF score.

# Overall Aim:
# This program demonstrates:
# Natural Language Processing (NLP)
# Tokenization
# POS Tagging
# Stopword Removal
# Stemming
# Lemmatization
# TF-IDF Representation
# and text feature extraction for machine learning.
