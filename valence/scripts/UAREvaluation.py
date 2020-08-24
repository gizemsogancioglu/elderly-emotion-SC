from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.model_selection import PredefinedSplit
import joblib
import pandas as pd
from valence.scripts import valence_classifier
from collections import Counter

def majority(arr):
    # convert array into dictionary
    freqDict = Counter(arr)
    # traverse dictionary and check majority element
    size = len(arr)
    major_key = None
    major_val = 1
    for (key, val) in freqDict.items():
        if (val > (size / 2)):
            return key
        else:
            if val > major_val:
                major_key = key
                major_val = val
            elif val == major_val:
                major_key = None
    return major_key

def majority_voting_pred(predictions, weight_index=None):
    i = 0
    voted_preds = []
    len_pred = len(predictions[0])
    while i < len_pred:
        pred_arr = list(map(lambda x: x[i], predictions))
        mv = majority(pred_arr)
        default_val = 'M'
        if mv is None:
            if weight_index is not None:
                mv = predictions[weight_index][i]
            # Medium is chosen if each classifier predicts a different label
            else:
                mv = default_val
        voted_preds.append(mv)
        i += 1
    return voted_preds

def maximum(a, b, c):
    list = [a, b, c]
    return max(list)

def ensemble_different_models(features, features_descr, y):
    clf_pred = []
    majority_pred = {}
    num_models = 3
    f = 0
    i = 0
    for exp in ["dev", "test"]:
        x_index = 1 if exp == "dev" else 0
        y_test = y[1] if exp == 'dev' else None
        while f < len(features_descr):
            while i < num_models:
                clf = joblib.load("data/models/" + features_descr[f] + "_fold" + str(i) + ".pkl")
                clf_pred.append(clf.predict(features[f][x_index]))
                i += 1
            i = 0
            hard_pred = majority_voting_pred(clf_pred)
            if exp == "dev":
                print(
                    "DEVEL SCORE: of the majority voting of 3 models that are trained on different set for the feature ",
                    features_descr[f], evaluate(hard_pred, y_test))
            majority_pred[features_descr[f]] = hard_pred
            clf_pred = []
            f += 1
        ensemble_pred = majority_voting_pred([majority_pred["ft_polarity"], majority_pred["bows"], majority_pred["dict"]],
                                           weight_index=0)
        if exp == "dev":
            pd.DataFrame(ensemble_pred).to_csv("data/predictions/fold4_dev_predictions.csv")
            print("DEVEL SCORE: (Majority voting) score of the ensemble model (Fasttext+Polarity - TFIDF - Dictionary) "
                  "on blind set: ", evaluate(ensemble_pred, y_test))
        else:
            pd.DataFrame(ensemble_pred).to_csv("data/predictions/test_predictions.csv")
    return ensemble_pred

def tune_on_devset(X_train, y_train, X_devel, y_devel):
    uar_scores = []
    # score representations with SVM on different complexity levels
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 0.20, 0.25, 0.30, 0.58, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.5, 2, 10]
    kernels = ["sigmoid", "linear", "rbf", "poly"]
    gamma_val = [1e-1, 1, 1e1, 'scale']
    kernel_dict = {}
    complexity_dict = {}
    gamma_dict = {}
    best_train_uar = {}
    for k in kernels:
        for g in gamma_val:
            for c in complexities:
                clf = SVC(C=c, random_state=0, kernel=k, gamma=g, probability=True)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_devel)
                UAR_score = evaluate(y_pred, y_devel)
                uar_scores.append(UAR_score)
                y_train_pred = clf.predict(X_train)
                best_train_uar[str(UAR_score)] = evaluate(y_train_pred, y_train)
                kernel_dict[uar_scores[-1]] = k
                complexity_dict[uar_scores[-1]] = c
                gamma_dict[uar_scores[-1]] = g
    UAR_dev = max(uar_scores)
    print("UAR dev is", UAR_dev)
    train_UAR = best_train_uar[str(UAR_dev)]
    print("UAR train is", train_UAR)
    best_kernel = kernel_dict[UAR_dev]
    best_comp = complexity_dict[UAR_dev]
    best_gamma = gamma_dict[UAR_dev]
    print("best kernel and best comp pair: ", best_kernel, best_comp, best_gamma)
    clf = SVC(C=best_comp, random_state=0, kernel=best_kernel, gamma=best_gamma, probability=True)
    clf.fit(X_train, y_train)
    return clf

def evaluate(y_pred, y):
    # Evaluation
    return recall_score(y, y_pred, average='macro') * 100

def k_fold_cv(X, y, feature_desc):
    # since fold 4 will be used as a blind set and not part of training, it is removed from fold_ids list.
    fold_ids = pd.read_csv("data/raw_data/CV_fold_ids_trval.csv")['FoldID'][0:132]
    ps = PredefinedSplit(fold_ids)
    fold_id = 0
    y = y[valence_classifier.label_type]
    for train_index, test_index in ps.split():
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = tune_on_devset(X_train, y_train, X_test, y_test)
        joblib.dump(clf, "data/models/" + feature_desc + "_fold" + str(fold_id) + '.pkl')
        fold_id += 1
    return
