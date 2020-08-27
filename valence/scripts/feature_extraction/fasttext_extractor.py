import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
import pandas as pd

def write_fasttext_features(df, file_path):
    #load fine-tuned fasttext model.
    ft = FastText.load('data/models/improved_fasttext_model')
    X_vector = normalize_text_fasttext(df, ft)
    df2 = pd.DataFrame(X_vector)
    print(df2.shape)
    df2.to_csv(file_path, index=False)
    return

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
