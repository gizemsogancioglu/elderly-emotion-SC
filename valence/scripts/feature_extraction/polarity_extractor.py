import flair
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
nltk.download('punkt')

sid = []
flair_sentiment = []

def read_data(filename, delim):
  df = pd.read_csv(filename, delimiter=delim)
  return df

def load_analyzer():
  nltk.download('vader_lexicon')
  sid = SentimentIntensityAnalyzer()
  flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
  return (sid, flair_sentiment)

(sid, flair_sentiment) = load_analyzer()

def nltk_get_sentiment(sentence):
    return sid.polarity_scores(sentence)

def textblob_get_sentiment(sentence):
    # we can use subjectivity analysis from textblob result since polarity prediction is not as good as NLTK.
    return TextBlob(sentence).sentiment

def flair_get_sentiment(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    s = ' '.join(map(str, total_sentiment))
    prob = s[s.find("(") + 1:s.find(")")]
    if "NEGATIVE" in s:
        prob = 1-abs(float(prob))
    return (prob)

# Polarity Feature Extraction per Story
def write_sentiment_features(df, filename):
    df_final = pd.DataFrame()
    sent_arr = get_sentiment_features_per_story(df)
    polarity_features = np.asarray(sent_arr)
    df_final["polarity"] = polarity_features[:, 0]
    df_final["pos"] = polarity_features[:, 1]
    df_final["neg"] = polarity_features[:, 2]
    df_final["neu"] = polarity_features[:, 3]
    df_final["compound"] = polarity_features[:, 4]
    df_final["subjectivity"] = polarity_features[:, 5]
    df_final["flair_prob"] = polarity_features[:, 6]
    df_final["flair_label"] = polarity_features[:, 7]

    df_final.to_csv(filename+"polarity_features.csv", index=False)

def compute_all_sent_features(text):
    sentiment = nltk_get_sentiment(text)
    textblob_sent = textblob_get_sentiment(text)
    (flair_prob) = flair_get_sentiment(text)
    return ([textblob_sent[0], sentiment["pos"], sentiment["neg"],
             sentiment["neu"], sentiment["compound"], textblob_sent[1], flair_prob])

def get_sentiment_features_per_story(X):
    story_arr = []
    i = 0
    for story in X:
        if story is not None:
          story_arr.append(compute_all_sent_features(story))
          i = i + 1
          print("story", i)
        else:
            story_arr.append([0, 0, 0, 0, 0, 0, 0, 0])
    return story_arr


