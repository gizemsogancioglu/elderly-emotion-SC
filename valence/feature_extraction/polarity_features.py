#Installation
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
    label = "positive"
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    s = ' '.join(map(str, total_sentiment))
    prob = s[s.find("(") + 1:s.find(")")]
    if "NEGATIVE" in s:
        # prob = -abs(float(prob))
        label = "negative"
    return (prob, label)

# Polarity Feature Extraction per Story
def write_sentiment_features(df, filename, label):
    df_final = pd.DataFrame()
    if label == "story-level":
      sent_arr = get_sentiment_features_per_story(df)
    else :
      sent_arr = get_sentiment_features_per_sentence(df)
    polarity_features = np.asarray(sent_arr)
    df_final["polarity"] = polarity_features[:, 0]
    df_final["pos"] = polarity_features[:, 1]
    df_final["neg"] = polarity_features[:, 2]
    df_final["neu"] = polarity_features[:, 3]
    df_final["compound"] = polarity_features[:, 4]
    df_final["subjectivity"] = polarity_features[:, 5]
    df_final["flair_prob"] = polarity_features[:, 6]
    df_final["flair_label"] = polarity_features[:, 7]

    if label == "story-level":
        df_final.to_csv(filename+"_per_story.csv", index=False)
    else:
        df_final.to_csv(filename+"_per_sentence.csv", index=False)

def compute_all_sent_features(sentence):
    sentiment = nltk_get_sentiment(sentence)
    textblob_sent = textblob_get_sentiment(sentence)
    (flair_prob, flair_label) = flair_get_sentiment(sentence)
    return ([textblob_sent[0], sentiment["pos"], sentiment["neg"],
             sentiment["neu"], sentiment["compound"], textblob_sent[1], flair_prob, flair_label])

def get_sentiment_features_per_story(X):
    story_arr = []
    i = 0
    for story in X:
        if story != None:
          story_arr.append(compute_all_sent_features(story))
          print("Story number : ", i)
          i = i + 1
        else: story_arr.append([0,0,0,0,0,0,0,0])
    return story_arr


def get_sentiment_features_per_sentence(X):
    story_arr = []
    flair_arr = []
    label_arr = []
    textblob_polarity = []
    pos = []
    neg =[]
    neu = []
    compound = []
    textblob_subject = []
    i = 0
    for story in X:
        for sent in nltk.sent_tokenize(story):
            (flair_prob, flair_label) = flair_get_sentiment(sent)
            sentiment = nltk_get_sentiment(sent)
            textblob_sent = textblob_get_sentiment(sent)
            textblob_polarity.append(textblob_sent[0])
            textblob_subject.append(textblob_sent[1])
            pos.append(sentiment["pos"])
            neg.append(sentiment["neg"])
            neu.append(sentiment["neu"])
            compound.append(sentiment["compound"])
            flair_arr.append(flair_prob)
            label_arr.append(flair_label)
        story_arr.append([textblob_polarity, pos, neg, neu, compound, textblob_subject, flair_arr, label_arr])
        flair_arr = []
        label_arr = []
        pos = []
        neg =[]
        neu = []
        compound =[]
        textblob_subject=[]
        textblob_polarity =[]
        print("story :", i)
        i = i + 1
    return story_arr