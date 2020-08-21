from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

def set_vectorizer(text_f):
    vectorizer = get_tfidf_vector(text_f)
    return vectorizer

def get_tfidf_vector(X_all):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), tokenizer=tokenize_text)
    vectorizer.fit_transform(X_all)
    return vectorizer

def tokenize_text(text):
    tokens = []
    stemmer = PorterStemmer()
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    stems = [stemmer.stem(item) for item in tokens]
    return stems

def normalize_text_data(X):
    return (set_vectorizer(X).transform(X))

def write_TFIDF_features(df, file_path):

    X_vector = normalize_text_data(df)
    df2 = pd.DataFrame(X_vector.toarray())
    df2.to_csv(file_path, index=False)
    return df2
