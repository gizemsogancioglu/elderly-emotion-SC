import pandas as pd
from valence.scripts.feature_extraction import tfidf_extractor, polarity_extractor, fasttext_extractor
from pathlib import Path
from valence.scripts import UAREvaluation
import os
from sklearn.preprocessing import StandardScaler
label_type = 'V_cat'
dict_selected_features = ['max_pos_SentiWordNet',
                     'sum_neg_SentiWordNet',
                     'min_SentiWS',
                     'max_SentiWS',
                     'num_neg_SentiWS']

def combine_feature_vectors(X_f1, X_f2):
    return pd.concat(
        [X_f1, X_f2], axis=1)

def extract_features():
    ### COMPUTE TFIDF Feature Vectorizer ###
    features = []
    feature_descr = ['ft_polarity', 'bows', 'dict']
    source_path = "source/features/"
    if not os.path.isfile(source_path+'fasttext_features.csv'):
        print("Fasttext features are not in the expected source/features folder.. We will extract them now.. ")
        ### Fasttext Feature Extractor ###
        fasttext_extractor.write_fasttext_features(raw_data["machine_translation"],
          source_path+ "fasttext_features.csv")

    fasttext_features = pd.concat([raw_data[['partition', 'FoldID']], pd.read_csv(
        source_path+ "fasttext_features.csv",
        sep=",").reset_index(drop=True)],
                                  axis=1)
    fasttext = fill_partitions(fasttext_features)

    if not os.path.isfile(source_path+'polarity_features.csv'):
        print("Polarity features are not in the expected source/features folder.. We will extract them now.. ")
        polarity_extractor.write_sentiment_features(raw_data['machine_translation'],
                                                    source_path, 'story-level')

    polarity_features = pd.concat([raw_data[['partition', 'FoldID']],
                                   pd.read_csv(source_path+"polarity_features.csv",sep=",").
                                  reset_index(drop=True)], axis=1).drop(columns='ID_story')
    polarity = fill_partitions(polarity_features)
    polarity = normalize_data(polarity)

    ft_polarity = []
    for i in [0, 1, 2]:
        ft_polarity.append(
            combine_feature_vectors(fasttext[i].reset_index(drop=True), polarity[i].reset_index(drop=True)))

    features.append(ft_polarity)

    if not os.path.isfile(source_path+'TFIDF_features.csv'):
        print("TFIDF features are not in the expected source/features folder.. We will extract them now.. ")
        tfidf_extractor.write_TFIDF_features(raw_data['machine_translation'], source_path+"TFIDF_features.csv")
    bows_features = pd.concat(
        [raw_data[['partition', 'FoldID']], pd.read_csv(source_path+"TFIDF_features.csv", sep=",").
            reset_index(drop=True)], axis=1)
    bows = fill_partitions(bows_features)

    features.append(bows)

    if os.path.isfile(source_path+'sentiWS_features.csv'):
        ## Dictionary Features
        features_1 = pd.read_csv(source_path+'sentiWS_features.csv')
        features_2 = pd.read_csv(source_path+'sentiWordNet_features.csv')
        dict_features = features_1.join(features_2.set_index('ID_story'), on='ID_story')[dict_selected_features].\
            reset_index(drop=True)
        dict_features = pd.concat([raw_data[['partition', 'FoldID']], dict_features], axis=1)
        dict = normalize_data(fill_partitions(dict_features))

    else:
        print("Dictionary features are not in the expected source/features folder.. We will extract them now.. ")
    features.append(dict)

    return (features, feature_descr)

def train_models(features, features_descr):
    file_list = os.listdir("source/models")
    i = 0
    while i < len(features):
        num_models = [filename for filename in file_list if filename.startswith(features_descr[i])]
        if len(num_models) != 3:
            print(features_descr[i] + " models are not located under the source/models folder.. Training starts.. ")
            UAREvaluation.k_fold_cv(features[i][0], y[0], features_descr[i])
        else:
            print(features_descr[i] + " models are already located in the source/models folder.")
        i += 1
    return

def normalize_data(X):
    scaler = StandardScaler()
    scaler = scaler.fit(X[0])
    # scale train/dev/test partitions
    for i in [0, 1, 2]:
        X[i] = pd.DataFrame(scaler.transform(X[i]))
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

#def translate_transcriptions():

if __name__ == "__main__":
    # Read raw data and predefined fold ids assigned per story.
    fold_ids = pd.read_csv("source/data/CV_fold_ids_trval.csv").reset_index(drop=True)
    raw_data = pd.concat([pd.read_excel(Path('source/data/data.xlsx'), index_col=0).reset_index(drop=True), fold_ids], axis=1)

    # translate transcriptions into English
    # translate_transcriptions()

    fold_id = 4
    y_labels = raw_data[[label_type, 'partition', 'FoldID']]
    y = fill_partitions(y_labels)

    (features, features_descr) = extract_features()
    print("Features are loaded..")
    train_models(features, features_descr)
    print("Bows-Fasttext+Polarity-Dictionary models are ready for the batch predictions")
    UAREvaluation.ensemble_different_models(features, features_descr, y)
