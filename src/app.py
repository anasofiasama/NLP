# Importing and installing librarys and packages
import pandas as pd
import regex as reg
import re
import matplotlib.pyplot as plt
import unicodedata
import nltk


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordsegment import load, segment

## Step 1
# Load the data
url='https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv'
df_raw = pd.read_csv(url)

df_interim=df_raw.copy() # copy of df_raw to make the processing

# Change target variable data type
df_interim['is_spam']=df_interim['is_spam'].astype('category')

# Encoding
df_interim['is_spam']=df_interim['is_spam'].cat.codes

## Step 2
# drop doplicate rows
df_interim.drop_duplicates(inplace=True)
df_interim.reset_index(inplace=True)

# change '-' for space
df_interim['url'] = df_interim['url'].str.replace(r'-', ' ', regex=True)

df_interim['url']=df_interim['url'].str.strip() # delet the firt and last space
df_interim['url']=df_interim['url'].str.lower() # change all the text to lower
# clean up strange symbols and standardize 
df_interim["url"] = df_interim["url"].apply(lambda x: re.sub(r'www'," ",x))
df_interim['url'] = df_interim['url'].str.replace('''[?&#,;ü_=%./']''','',regex=True)
df_interim['url'] = df_interim['url'].str.replace(r'http(s):', '', regex=True)
df_interim['url'] = df_interim['url'].str.replace(r'http:', '', regex=True)
df_interim['url'] = df_interim['url'].str.replace(r'com', '', regex=True)
# remove all numbers
df_interim['url'] = df_interim['url'].str.replace(r'[\d-]', '', regex=True) 
# remove all single letters
df_interim["url"] = df_interim["url"].apply(lambda x: re.sub(r'\b[a-zA-Z]\b'," ",x))
# remove characters repited more than two times
df_interim['url']=df_interim['url'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True)

# remove stopwords
stop=stopwords.words('english')
# function to remove stopwords
def remove_stopwords(message):
    if message is not None:
        words = message.strip().split()
        words_filtered = []
        for word in words:
            if word not in stop:
                words_filtered.append(word)
                result = " ".join(words_filtered)
    else:
        result = None

    return result

df_interim['url']=df_interim['url'].apply(remove_stopwords)

# words segmentation
load() # reads and parses the unigrams and bigrams data from disk
# Function that makes words segmentarion to each row
def segment_words(message):
    if message is not None:
        words = segment(message)
        words_filtered = []
        for word in words:
            if word not in stop: # if any word is in stopwords, it removes
                words_filtered.append(word)
                result = " ".join(words_filtered)
    else:
        result = None

    return result

df_interim['url']=df_interim['url'].apply(segment_words)

# copy to the final dataset
df_final=df_interim.copy()

# define target and feature variable
y=df_final['is_spam']
X=df_final['url']

# split dataset between training and testing sample
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2007,stratify=y)

# tokenization
vec=CountVectorizer(stop_words='english')
X_train=vec.fit_transform(X_train).toarray()
X_test=vec.transform(X_test).toarray()

# SVM clasiffier
clf_svm = svm.SVC(C=1.0, kernel='linear', degree=3, random_state=3107,gamma='auto',class_weight='balanced') 

clf_svm.fit(X_train, y_train) 
y_pred = clf_svm.predict(X_test)
print('Accuracy scores')
print(classification_report(y_test, y_pred))

import pickle
filename = '/workspace/NLP/models/finalized_model.sav'
pickle.dump(clf_svm, open(filename, 'wb'))