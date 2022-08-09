import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models.word2vec import Word2Vec
from collections import Counter

def getMostFrequentWords(frequency_threshold, texts):
    ##### This code creates a list of words above the preset frequency threshold
    frequent_keywords = []

    ####  We first will collect all tokens from all the utterances from the training data using the NLTL tokenizer
    alltokens = []
    for text in texts:
        tokenlist = nltk.tokenize.word_tokenize(text)
        for token in tokenlist:
            alltokens.append(token)
        
    #### The Counter function will create a frequency count of all the items. The result is a dictionary       
    kw_counter = Counter(alltokens)

    #### We now loop over the dictionary with counts to get the word and the frequency value
    for word, count in kw_counter.items():
        if count>frequency_threshold:
            frequent_keywords.append(word)
    
    print('Nr of words above the frequency threshold', len(frequent_keywords))
    print('Frequency threshold', frequency_threshold)
    return frequent_keywords

# Function to average all word vectors in a paragraph
def featureVecMethod(list_of_words_in_the_utterance, # Tokenized list of tokens from an utterance
                     frequent_keywords, # List of words above the frequency threshold
                     stop_words, # Stopwords that should be skipped
                     model, # The actual word embeddings model
                     modelword_index, # An index on the vocabulary of the model to speed up lookup
                     num_embedding_dimensions # the number of dimensions of the embedding model
                    ):
    # Pre-initialising empty numpy array (np) for speed
    # This create a numpy array with the length of the num_features set to zero values
    featureVec = np.zeros(num_embedding_dimensions,dtype="float32")
    
    ## A counter for the number of tokens represented so that we can take the average
    nwords = 0
    embedding_words= []
    no_embedding_words= []
    
    for word in  list_of_words_in_the_utterance:
        #### we only use words that are frequent and not stopwords
        if word in frequent_keywords and not word in stop_words:         
            if word in modelword_index:
                ## The next function would directly take the values from the model
                #featureVec = np.add(featureVec,model[word])
                
                ### Instead of simply taking the embedding values we prefer the next function that 
                ### normalises these values between [-1, 1] to make the average work better
                ### Don't worry about the specifics 
                featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))

                #we keep track of the words detected, just for analysing the data
                embedding_words.append(word)
                nwords = nwords + 1
            else:
                word = word.lower()
                if word in modelword_index:
                    #featureVec = np.add(featureVec,model[word])
                    featureVec = np.add(featureVec,model[word]/np.linalg.norm(model[word]))
                    
                    #we keep track of the words detected, just for analysing the data
                    embedding_words.append(word)
                    nwords = nwords + 1
                else:
                    #we keep track of the unknown words to see how well our model fits the data
                    no_embedding_words.append(word)
                    
    # Dividing the result by number of words in each utterance to get average
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    
    #Our function returns 3 results
    return featureVec, embedding_words, no_embedding_words


# Function for calculating the average feature vector
def getAvgFeatureVecs(texts, ### List of texts, in our case tokenized utterances
                      keywords, 
                      stopwords, 
                      model, 
                      modelword_index, 
                      num_embedding_dimensions
                     ):
    counter = 0
    embedding_words=[]
    no_embedding_words=[]

    #### we initialise a numpy matrix with the number of rows defined by the lengths of the texts (all utterances) and the columns the number of dimensions. 
    #### All cells will have zeros of the type float32
    textFeatureVecs = np.zeros((len(texts),num_embedding_dimensions),dtype="float32")
    print('Shape of our matrix is:',textFeatureVecs.shape)
    
    #### We iterate over all the texts
    for text in texts:
        # Printing a status message every 1000th text, to see what we are processing
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(texts)))
        ### Each text is transformed into a vector representation based on the averaged token embedding using the previous function
        ### We add these vectors to the total matrix
        textFeatureVecs[counter], emb_words, nemb_words = featureVecMethod(text, 
                                                                           keywords, 
                                                                           stopwords, 
                                                                           model, 
                                                                           modelword_index,
                                                                           num_embedding_dimensions)
        counter = counter+1
        #print(emb_words)
        embedding_words.extend(emb_words)
        #print(embedding_words)
        no_embedding_words.extend(nemb_words)
        
    #### Due to the averaging, there could be infinitive values or NaN values. The next numpy function turns these value to "0" scores
    textFeatureVecs = np.nan_to_num(textFeatureVecs) 
    
    return textFeatureVecs, embedding_words, no_embedding_words



### Use a mapping to get a dictionary of the mapped GO_emotion scores
def get_mapped_scores(emotion_map, go_emotion_scores):
    mapped_scores = {}

    for prediction in go_emotion_scores[0]:
        go_emotion=prediction['label']
        for key in emotion_map:
            if go_emotion in emotion_map[key]:
                if not key in mapped_scores:
                    mapped_scores[key]= [prediction['score']]
                else:
                    mapped_scores[key].append(prediction['score'])
    return mapped_scores

### Get the averaged score for an emotion or sentiment from the GO_emotion scores mapped according to the emotion_map
def get_averaged_mapped_scores(emotion_map, go_emotion_scores):
    averaged_mapped_scores = []
    mapped_scores = get_mapped_scores(emotion_map, go_emotion_scores)
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