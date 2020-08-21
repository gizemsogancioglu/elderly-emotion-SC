import fasttext.util
import numpy as np
import nltk
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
import pandas as pd

#pretrained model, retrained with the ESC dataset.
#ft_de = load_facebook_model('/Users/Gizem/workspace/projects/esc_resources/embeddings/cc.de.100.bin')

def write_fasttext_features(df, file_path):
    ft = FastText.load('/Users/Gizem/PycharmProjects/compare-esc/embeddings/MOOD_improved_fasttext_model')
    X_vector = normalize_text_fasttext(df, ft)
    df2 = pd.DataFrame(X_vector)
    print(df2.shape)
    df2.to_csv(file_path, index=False)
    return

def retrain_fasttext_model(data, ft_model):
    word_arr = []
    sent_arr = []
    for story in data:
        for sent in nltk.sent_tokenize(story):
            for word in word_tokenize(sent):
                word_arr.append(word)
            sent_arr.append(word_arr)
            word_arr = []
    ft_model.build_vocab(sent_arr, update=True)
    ft_model.train(sentences=sent_arr, total_examples=len(sent_arr), epochs=5)
    ft_model.save('improved_fasttext_model')

def normalize_text_fasttext(data, ft):
    result_arr = []
    for instance in data:
        ins_vec = list(map(lambda x: ft.wv[x.lower()], word_tokenize(instance)))
        result_arr.append(np.mean(ins_vec, axis=0, dtype=np.float64))
    res_vector = np.array(result_arr)
    return res_vector

def normalize_data(X_train, X_devel):
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_devel = scaler.transform(X_devel)
    return (X_train, X_devel)

def load_ft_model(lang, dim):
    embedding_name = "cc."+lang+"."+dim+".bin"
    ft = fasttext.load_model('embeddings/'+embedding_name)
    return ft
