# =========================================================
# pip install nltk numpy pandas matplotlib seaborn scikit-learn
# Install required libraries for NLP and visualization

# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# Downloads required NLTK resources (run once)
# =========================================================

import nltk  # Natural Language Processing library
import numpy as np  # Numerical operations
import pandas as pd  # Data handling
import matplotlib.pyplot as plt  # Graph plotting
import seaborn as sns  # Advanced visualization

from nltk.tokenize import word_tokenize

# Used for splitting text into words/tokens

from nltk.corpus import stopwords

# Provides stopword list like is, the, and etc.

from nltk.stem import PorterStemmer, WordNetLemmatizer

# PorterStemmer -> stemming
# WordNetLemmatizer -> lemmatization

from nltk import pos_tag

# Used for Part-of-Speech tagging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# CountVectorizer -> TF calculation
# TfidfVectorizer -> TF-IDF calculation


# =========================================================
# DOWNLOAD NLTK DATA (RUN ONCE)
# =========================================================

nltk.download("punkt_tab")
# Downloads tokenizer table data

nltk.download("punkt")
# Downloads tokenizer package

nltk.download("stopwords")
# Downloads stopword dataset

nltk.download("averaged_perceptron_tagger")
# Downloads POS tagging model

nltk.download("wordnet")
# Downloads WordNet for lemmatization

nltk.download("averaged_perceptron_tagger_eng")
# Downloads English POS tagger


# =========================================================
# SAMPLE DOCUMENT
# =========================================================

text = """Data science is an interdisciplinary field that uses scientific methods,
processes, algorithms and systems to extract knowledge and insights from data."""
# Sample text/document for NLP operations

print("\nOriginal Text:\n", text)
# Output: Displays original input text


# =========================================================
# 1. TOKENIZATION
# =========================================================

tokens = word_tokenize(text)
# Splits sentence into individual words/tokens

print("\nTokens:\n", tokens)
# Output: Displays list of tokens


# =========================================================
# 2. POS TAGGING
# =========================================================

pos_tags = pos_tag(tokens)
# Assigns grammatical tags to each token

print("\nPOS Tags:\n", pos_tags)
# Output: Displays words with grammatical tags
# Example:
# NN -> Noun
# VB -> Verb
# JJ -> Adjective


# =========================================================
# 3. STOPWORD REMOVAL
# =========================================================

stop_words = set(stopwords.words("english"))
# Loads English stopwords

filtered_words = [word for word in tokens if word.lower() not in stop_words]
# Removes common unnecessary words like:
# is, an, and, the etc.

print("\nAfter Stopword Removal:\n", filtered_words)
# Output: Displays meaningful words after removing stopwords


# =========================================================
# 4. STEMMING
# =========================================================

stemmer = PorterStemmer()
# Creates stemming object

stemmed_words = [stemmer.stem(word) for word in filtered_words]
# Reduces words to root/base form

print("\nStemmed Words:\n", stemmed_words)
# Output: Displays stemmed words
# Example:
# learning -> learn


# =========================================================
# 5. LEMMATIZATION
# =========================================================

lemmatizer = WordNetLemmatizer()
# Creates lemmatizer object

lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
# Converts words into meaningful dictionary root form

print("\nLemmatized Words:\n", lemmatized_words)
# Output: Displays lemmatized words


# =========================================================
# 6. TF CALCULATION
# =========================================================

documents = [
    text,
    "Data science involves machine learning and data analysis",
    "Machine learning is a part of data science",
]
# List of sample documents

count_vectorizer = CountVectorizer()
# Creates CountVectorizer object

tf_matrix = count_vectorizer.fit_transform(documents)
# Calculates Term Frequency matrix

print("\nTerm Frequency (TF):\n", tf_matrix.toarray())
# Output: Displays frequency of words in documents


# =========================================================
# 7. IDF CALCULATION
# =========================================================

tfidf_vectorizer = TfidfVectorizer()
# Creates TF-IDF vectorizer object

tfidf_vectorizer.fit(documents)
# Learns IDF values from documents

idf_values = tfidf_vectorizer.idf_
# Extracts IDF values

print("\nInverse Document Frequency (IDF):\n", idf_values)
# Output: Displays importance score of words


# =========================================================
# 8. TF-IDF MATRIX
# =========================================================

tfidf_matrix = tfidf_vectorizer.transform(documents)
# Converts text into TF-IDF matrix

print("\nTF-IDF Matrix:\n", tfidf_matrix.toarray())
# Output: Displays TF-IDF scores for all words


# =========================================================
# 📊 VISUALIZATION FIX (FULL WORDS VISIBLE)
# =========================================================

feature_names = count_vectorizer.get_feature_names_out()
# Extracts all unique words/features

tf_df = pd.DataFrame(tf_matrix.toarray(), columns=feature_names)
# Converts TF matrix into DataFrame

tf_sum = tf_df.sum().sort_values(ascending=False)
# Calculates total frequency of each word

plt.figure(figsize=(14, 6))
# Creates graph with larger width

tf_sum.plot(kind="bar")
# Bar chart of term frequencies

plt.title("Term Frequency Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.xticks(rotation=45, ha="right")
# Rotates labels for better visibility

plt.tight_layout()
# Prevents labels from getting cut

plt.show()
# Displays TF graph


# TF-IDF plot fix
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
)
# Converts TF-IDF matrix into DataFrame

tfidf_sum = tfidf_df.sum().sort_values(ascending=False)
# Calculates total TF-IDF score for each word

plt.figure(figsize=(14, 6))

tfidf_sum.plot(kind="bar")
# Bar chart of TF-IDF importance scores

plt.title("TF-IDF Importance of Words")
plt.xlabel("Words")
plt.ylabel("Score")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
# Displays TF-IDF graph


# ============================================================
# COMPLETE OUTPUT EXPLANATION
# ============================================================

# 1. Original Text Output
# Displays the original input text/document used for NLP tasks.

# 2. Tokenization Output
# Splits sentence into individual words/tokens.

# Example:
# ['Data', 'science', 'is', 'an', ...]

# Tokenization is first step in NLP preprocessing.

# 3. POS Tagging Output
# Displays each word with grammatical tag.

# Example:
# ('Data', 'NN')
# NN -> Noun
# VB -> Verb
# JJ -> Adjective

# Helps understand grammatical role of words.

# 4. Stopword Removal Output
# Removes common words like:
# is, an, the, and etc.

# Output contains only meaningful words.
# Improves NLP processing efficiency.

# 5. Stemming Output
# Converts words into root form.

# Example:
# learning -> learn
# processes -> process

# Stemmed words may not always be meaningful dictionary words.

# 6. Lemmatization Output
# Converts words into proper dictionary root form.

# Example:
# studies -> study
# running -> run

# Lemmatization gives more meaningful results than stemming.

# 7. TF (Term Frequency) Output
# Displays frequency of words in each document.

# Example:
# If "data" appears 3 times,
# its frequency value becomes higher.

# TF helps identify commonly used words.

# 8. IDF (Inverse Document Frequency) Output
# Measures importance of words across documents.

# Common words get lower IDF score.
# Rare/important words get higher IDF score.

# 9. TF-IDF Matrix Output
# Combines TF and IDF values.

# Formula:
# TF-IDF = TF × IDF

# High TF-IDF score means:
# Word is important in a document
# but not common in all documents.

# 10. Term Frequency Graph Output
# Bar graph showing frequency of words.
# Taller bars represent frequently occurring words.

# 11. TF-IDF Graph Output
# Displays importance score of words.
# Higher bars indicate more important words.

# Overall Aim:
# This program demonstrates:
# Natural Language Processing (NLP)
# Tokenization
# POS Tagging
# Stopword Removal
# Stemming
# Lemmatization
# TF Calculation
# IDF Calculation
# TF-IDF Analysis
# and visualization of word importance.
