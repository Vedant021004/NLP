from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

word = input("Enter a word: ")

print("Lemma:", lemmatizer.lemmatize(word, pos="v"))