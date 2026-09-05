from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# User input
query = input("Enter sentence: ")


# 1. Tokenization
tokens = word_tokenize(query)

print("\nTokens:", tokens)


# 2. Stopwords
stopword = stopwords.words("english") # language ko define krna pdta hai

print("\nStopwords:", stopword)


# 3. Stemming
user = input("\nEnter a word for stemming: ")

stemmer = PorterStemmer()

print("Stemming:", stemmer.stem(user))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

word = input("Enter a word: ")

print("Lemma:", lemmatizer.lemmatize(word, pos="v"))