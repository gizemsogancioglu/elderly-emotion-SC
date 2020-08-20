import pandas as pd
from valence.feature_extraction import tfidf_extractor, fasttext_extractor, polarity_features
from pathlib import Path
from valence import UAREvaluation
import os
from sklearn.preprocessing import StandardScaler

label_type = 'V_cat'

def combine_feature_vectors(X_f1, X_f2):
    return pd.concat(
        [X_f1, X_f2], axis=1)

def extract_features():
    ### COMPUTE TFIDF Feature Vectorizer ###
    tfidf_extractor.write_TFIDF_features(raw_data['machine_translation'], "features/TFIDF_features.csv")
    ### COMPUTE Polarity Features ###
    for f in ["story-level", "sentence-level"]:
        polarity_features.write_sentiment_features(raw_data['machine_translation'],
                                                    "features/polarity", f)
    ### Fasttext Feature Extractor ###
    fasttext_extractor.write_fasttext_features_for_mood_data(raw_data["machine_translation"], "features/fasttext_features.csv")

# def train_uni_models():
#
#     features = [fasttext_train, bows_train,  ]
#     feat_descr = ["fasttext", ""]
#     UAREvaluation.k_fold_cv_ensembling(, y_train, "fasttext")

#def translate_transcriptions():

def normalize_data(X):
    scaler = StandardScaler()
    scaler = scaler.fit(X[0])
    # scale train/dev/test partitions
    for i in [0, 1, 2]:
        X[i] = scaler.transform(X[i])
    return X

def fill_partitions(df):
    df_new = []
    df_new.append(df.loc[(df['partition'] == 'train') | (df['partition'] == 'devel')].
                  drop(columns=['partition']))
    # Use Fold id 4 as blind set for the experiments
    # [0] - > (Training + Development set) - Fold 4
    # [1] -> Fold 4 (in dev set) as blind set
    # [2] -> Test partition
    df_new.append(df_new[0][df_new[0]['FoldID'] == fold_id].drop(columns=['FoldID']))
    df_new[0] = df_new[0].drop(df_new[1].index).drop(columns=['FoldID'])
    df_new.append(df.loc[df['partition'] == 'test'].drop(columns=['partition', 'FoldID']))
    return df_new

if __name__ == "__main__":
    # translate transcriptions into English
    #translate_transcriptions()
    fold_ids = pd.read_csv("data/CV_fold_ids_trval.csv").reset_index(drop=True)
    raw_data = pd.concat([pd.read_excel(Path('data/data.xlsx'), index_col=0).reset_index(drop=True), fold_ids], axis=1)
    #extract_features()
    #### READ FEATURE FILES ####
    fold_id = 4
    y_labels = raw_data[[label_type, 'partition', 'FoldID']]
    y = fill_partitions(y_labels)

    fasttext_features = pd.concat([raw_data[['partition', 'FoldID']], pd.read_csv("features/fasttext_features.csv",
                                                                                  sep=",").reset_index(drop=True)],
                                  axis=1)
    fasttext = fill_partitions(fasttext_features)

    # # TFIIDF MODEL
    bows_features = pd.concat([raw_data[['partition', 'FoldID']], pd.read_csv("features/TFIDF_features.csv", sep=",")
                              .reset_index(drop=True)], axis=1)
    bows = fill_partitions(bows_features)

    # # STORY-LEVEL POLARITY FEATURES
    polarity_features = pd.concat([raw_data[['partition', 'FoldID']], pd.read_csv("features/polarity_per_story.csv",
                                                                                  sep=",").
                                  reset_index(drop=True)], axis=1).drop(columns='ID_story')

    polarity = fill_partitions(polarity_features)
    polarity = normalize_data(polarity)

    # # Dictionary Features
    dict_features = pd.concat([raw_data[['partition', 'FoldID']], pd.read_csv("features/sentiWS_features.csv", sep=",")
                              .reset_index(drop=True)], axis=1).drop(columns='ID_story')
    dict = fill_partitions(dict_features)
    dict = normalize_data(dict)

    ft_polarity = []
    for i in [0, 1, 2]:
        ft_polarity.append(combine_feature_vectors(fasttext[i], polarity[i]))

    print("Cross Validation set shape", bows[0].shape, " ,(Fold 4) Blind set shape: ",
          bows[1].shape, " ,Test set shape", bows[2].shape)

    file_list = os.listdir("models")
    ft_polarity_models = [filename for filename in file_list if filename.startswith("ft_polarity")]
    if len(ft_polarity_models) != 3:
        UAREvaluation.k_fold_cv(ft_polarity[0], y[0], "ft_polarity")

    bows_models = [filename for filename in file_list if filename.startswith("bows")]
    if len(bows_models) != 3:
        UAREvaluation.k_fold_cv(bows[0], y[0], "bows")

    dict_models = [filename for filename in file_list if filename.startswith("dict")]
    if len(dict_models) != 3:
        UAREvaluation.k_fold_cv(dict[0], y[0], "dict")

    # ensemble_different_models(3)

