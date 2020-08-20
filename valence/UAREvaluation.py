from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.model_selection import  PredefinedSplit
from sklearn.externals import joblib
import pandas as pd
from valence import valence_classifier
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

def majority_voting_pred (predictions,  weight_index=None):
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
            #Medium is chosen if each classifier predicts a different label
            else:
                mv = default_val
        voted_preds.append(mv)
        i += 1
    return voted_preds

def ensemble_different_models(fold_number):
    majority_preds = preds_for_blind_test(fold_number)
    predictions = []
    y_test = valence_classifier.y[1]
    predictions.append(majority_preds["FT_polarity"])
    predictions.append(majority_preds["tfidf"])
    predictions.append(majority_preds["dict"])

    voted_preds = majority_voting_pred(predictions, weight_index=0)

    print("(Majority voting) across different models, on blind set with feature set (:", evaluate(voted_preds, y_test))

def maximum(a, b, c):
    list = [a, b, c]
    return max(list)

def preds_for_blind_test(fold_number):
    features_descr = ["FT_polarity", "tfidf",  "dict"]
    features = [valence_classifier.ft_polarity,  valence_classifier.bows, valence_classifier.dict]
    clf_pred = []
    majority_preds = {}
    y_test = valence_classifier.y[1]
    l = 0
    i = 0
    while l < len(features_descr):
        while i < fold_number:
            clf = joblib.load("models/" + features_descr[l] + "_fold" + str(i) + ".pkl")
            hard_label_pred = clf.predict(features[l][1])
            clf_pred.append(hard_label_pred)
            print("UAR score after CV for devel set with feature set for fold (:",
                  i, "(", features_descr[l], ")", evaluate(hard_label_pred, y_test))
            i += 1
        i = 0
        hard_preds = majority_voting_pred(clf_pred)
        majority_preds[features_descr[l]] = hard_preds
        print("(Majority voting) UAR score after CV, on devel set with feature set (:", features_descr[l],
           ")", evaluate(hard_preds, valence_classifier.y[1]))
        clf_pred = []
        l += 1
    return (majority_preds)

def tune_on_devset(X_train, y_train, X_devel, y_devel, X_test=None):
    uar_scores = []
    # score representations with SVM on different complexity levels
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 0.20, 0.25, 0.30, 0.58, 0.8, 0.9, 1, 1.1, 1.2,  1.4, 1.5, 2, 10]
    kernels = ["sigmoid", "linear", "rbf", "poly"]
    gamma_val = [1e-1, 1, 1e1, 'scale']
    kernel_dict = {}
    complexity_dict = {}
    gamma_dict = {}
    best_pred = {}
    train_pred = {}
    best_train_uar = {}
    for k in kernels:
        for g in gamma_val:
            for c in complexities:
                clf = SVC(C=c, random_state=0, kernel=k, gamma=g, probability=True)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_devel)
                UAR_score = evaluate(y_pred, y_devel)
                uar_scores.append(UAR_score)
                best_pred[str(UAR_score)] = y_pred
                y_train_pred = clf.predict(X_train)
                train_pred[str(UAR_score)] = y_train_pred
                best_train_uar[str(UAR_score)] = evaluate(y_train_pred, y_train)
                kernel_dict[uar_scores[-1]] = k
                complexity_dict[uar_scores[-1]] = c
                gamma_dict[uar_scores[-1]] = g
    UAR_dev = max(uar_scores)
    print("UAR dev is", UAR_dev)
    y_pred = best_pred[str(UAR_dev)]
    y_train_pred = train_pred[str(UAR_dev)]
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
    return (recall_score(y, y_pred, average='macro')* 100)

def k_fold_cv(X, y, feature_desc):
    # fold 1, 2, 3, 4 will be blind set respectively.
    fold_ids = pd.read_csv("data/CV_fold_ids_trval.csv")['FoldID']
    #for i in range(1, 5):
    # comment out if experiments for different set of folds are needed.

    # Fold 4 will be used as a blind set, while Fold 1+2+3 will be train set for cross-validation.
    # i = 4
    # y_test = y[y['FoldID'] == i][valence_classifier.label_type]
    # y = y.drop(y_test.index)[valence_classifier.label_type]
    # X_test = X.iloc[y_test.index].drop(columns=['FoldID'])
    # X = X.iloc[y.index].drop(columns=['FoldID'])
    fold_ids = fold_ids[0:132]
    ps = PredefinedSplit(fold_ids)
    id = 0
    y = y[valence_classifier.label_type]
    for train_index, test_index in ps.split():
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = tune_on_devset(X_train, y_train, X_test, y_test)
        joblib.dump(clf, "models/" +feature_desc + "_fold" + str(id) + '.pkl')
        id += 1
    return