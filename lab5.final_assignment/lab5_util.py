import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.word2vec import Word2Vec
from collections import Counter


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


### Given a Pandas dataframe "df" and a set of labels extracted from it
### this function will filter out the labels that do NOT have "Eliza" as the speaker
def remove_eliza_labels (df, labels:[]):
    human_labels = []
    speakers = df['speaker']
    for speaker, label in zip(speakers, labels):
        if not speaker=='Eliza':
            human_labels.append(label)
    return human_labels