import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.word2vec import Word2Vec
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_test_utterances_and_labels (df):
    test_instances =[]
    test_labels = []
    for index, utterance in enumerate(df['utterance']):
        if not df['speaker'].iloc[index]=='Eliza':
            if not pd.isnull(df['utterance'].iloc[index]):
                test_instances.append(utterance)
                gold = df['Gold'].iloc[index]
                test_labels.append(gold)
    return test_instances, test_labels
            

# Fixing encoding problems and replacing the 'Utterance' columns with the cleaned strings
def replace_weird_tokens_in_meld(df):
    weird = ["\x92","\x97","\x91","\x93","\x94","\x85"]
    utts = []
    for utterance in df['Utterance']:
        for w in weird:
            utterance = utterance.replace(w, "'")
        utts.append(utterance)
    df['Utterance'] = utts
    

def plot_labels_with_counts(labels, values):
    total = 0
    for v in values:
        total+=v
    print('Total of values', total)
    ax = sns.barplot(x=labels, y=values)
    # Add values above bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.2, str(int((v/total*100)))+'%', ha='center')
    plt.show()
    
### Use a mapping to get a dictionary of the mapped GO_emotion scores above the thereshold
def get_mapped_scores(emotion_map, go_emotion_scores, threshold):
    mapped_scores = {}

    for prediction in go_emotion_scores[0]:
        if prediction['score']>=threshold:
            go_emotion=prediction['label']
            for key in emotion_map:
                if go_emotion in emotion_map[key]:
                    if not key in mapped_scores:
                        mapped_scores[key]= [prediction['score']]
                    else:
                        mapped_scores[key].append(prediction['score'])
    return mapped_scores

### Get the averaged score for an emotion or sentiment from the GO_emotion scores above a threshold
### mapped according to the emotion_map
def get_averaged_mapped_scores_by_threshold(emotion_map, go_emotion_scores, threshold):
    averaged_mapped_scores = []
    mapped_scores = get_mapped_scores(emotion_map, go_emotion_scores, threshold)
    for emotion in mapped_scores:
        lst = mapped_scores[emotion]
        averaged_score= sum(lst)/len(lst)
        averaged_mapped_scores.append({'label':emotion, 'score':averaged_score})
    return sort_predictions(averaged_mapped_scores)

### Mapping GO_Emotions to sentiment values
sentiment_map={
"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
"negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
"ambiguous": ["realization", "surprise", "curiosity", "confusion"]
}

### Mapping GO_Emotions to Ekman values
ekman_map={
"anger": ["anger", "annoyance", "disapproval"],
"disgust": ["disgust"],
"fear": ["fear", "nervousness"],
"joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
"sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
"surprise": ["surprise", "realization", "confusion", "curiosity"],
"neutral": ["neutral"]
}

### Sort a list of results in JSON format by the value of the score element
def sort_predictions(predictions):
    return sorted(predictions, key=lambda x: x['score'], reverse=True)
