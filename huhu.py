#!/usr/bin/env python
# coding: utf-8

# # Depression detector by favourite Spotify Playlist

# In[2]:


import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
import time
from time import sleep


import streamlit as st
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold, train_test_split


from array import *
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from livelossplot import PlotLossesKeras

import math


import pandas as pd
from stqdm import stqdm

stqdm.pandas()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

es = EarlyStopping(monitor = 'loss', patience = 3)
#------------------------------------------------------------------------------------------------------------------------------#
client_id= '4e8a6305a50f41208b67294e49f5fd05'
client_secret='7b4f7d653d3b4c99b5dec2eda8a76f5e'
##Spotify Developer account provide client id and secret for usag in coding

#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

#user input playlist url
st.title("Welcome to Spotify Depression Detector!")
st.subheader("Want to know your depression percentage?")
playlist_URL=st.text_input ("Drop your playlist URL in the box below:")


playlist_URI = playlist_URL.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]

track_data = []

Valence = []
Danceability = []
Energy = []
Key = []
Liveness = []
Loudness = []
Speechiness = []
Tempo = []

for track in sp.playlist_tracks(playlist_URI)["items"]:
    #Track name
    track_name = track["track"]["name"]
    #print(track_name)

    #URI
    track_uri = track["track"]["uri"]
    track_data = sp.audio_features(track_uri)
    #Store data into array
    Valence.append(track_data[0]['valence'])
    Danceability.append(track_data[0]['danceability'])
    Energy.append(track_data[0]['energy'])
    Key.append(track_data[0]['key'])
    Liveness.append(track_data[0]['liveness'])
    Loudness.append(track_data[0]['loudness'])
    Speechiness.append(track_data[0]['speechiness'])
    Tempo.append(track_data[0]['tempo'])




dataset=pd.read_csv("musicmood.csv")

y = dataset[['mood']]
x = dataset[['valence','danceability','energy','key','liveness','loudness','speechiness','tempo']]

x2 = MinMaxScaler().fit_transform(x)

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,encoded_y,test_size = 0.2, random_state = 15)

target = pd.DataFrame({'mood':dataset['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],
                                                                                                          ascending=True)

X_Train = x_train.copy()
Y_Train = y_train.copy()

X_Test = x_test.copy()
Y_Test = y_test.copy()

numericColumns = x_train.columns.tolist()

scaler = MinMaxScaler()
scaler.fit(dataset[numericColumns])
X_Train[numericColumns] = scaler.transform(X_Train[numericColumns])
X_Test[numericColumns] = scaler.transform(X_Test[numericColumns])

Y_Train = pd.DataFrame(Y_Train)
Y_Train.columns = ['mood']
Y_Test = pd.DataFrame(Y_Test)
Y_Test.columns = ['mood']

X_Train.to_parquet('X_Train.parquet')
X_Test.to_parquet('X_Test.parquet')

pd.DataFrame(Y_Train).to_parquet('Y_Train.parquet')
pd.DataFrame(Y_Test).to_parquet('Y_Test.parquet')

# Load processed data from disk
X_Train = pd.read_parquet('X_Train.parquet')
X_Test = pd.read_parquet('X_Test.parquet')

Y_Train = pd.read_parquet('Y_Train.parquet')
Y_Test = pd.read_parquet('Y_Test.parquet')
#------------------------------------------------------------------------------------------------------------------------------#
#Create the model
def base_model():
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
    #Compile the model using sigmoid loss function and adam optim
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                  metrics=['accuracy'])
    return model

def predict_mood(valence, danceability, energy, key, liveness, loudness, speechiness, tempo):
    new_input = {'valence':valence,'danceability':danceability,'energy':energy,'key':key,
                 'liveness' :liveness,'loudness':loudness,'speechiness':speechiness,'tempo':tempo}
    new_input_df = pd.DataFrame([new_input])
    new_input_df = scaler.transform(new_input_df)
    X_new_input = new_input_df

    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=20,batch_size=250,
                                                                             verbose=0,callbacks=[es]))])
    #Fit the Pipeline
    pip.fit(x,encoded_y)
    #Predict the features of the song
    results = pip.predict(X_new_input)

    mood = np.array(target['mood'][target['encode']==int(results)])
    return mood[0]

#------------------------------------------------------------------------------------------------------------------------------#
#For reprinting loading progress
_last_print_len = 0
def reprint(msg,msg2,msg3, finish=False):
    global _last_print_len

    print(' '*_last_print_len, end='\r')

    if finish:
        end = '\n'
        _last_print_len = 0
    else:
        end = '\r'
        _last_print_len = len(msg)

    st.write(msg,msg2,msg3, end=end)
#------------------------------------------------------------------------------------------------------------------------------#
Song_mood = []
song_name = []
i = 0
j = len(Valence)
st.write("Please wait while we processing your playlist...")
my_bar = st.progress(0)

            
for track in sp.playlist_tracks(playlist_URI)["items"]:
        songmood = predict_mood(Valence[i], Danceability[i], Energy[i], Key[i],Liveness[i], Loudness[i], Speechiness[i], Tempo[i])
        Song_mood.append(songmood)
        song_name.append(track["track"]["name"])
        i = i + 1
        percent_complete=math.trunc((i/j)*100)
       
        my_bar.progress(percent_complete)
   
      
    
my_bar.progress(100)  


st.title("Proccessing complete!  ")
#calculate percentage of sad song in the playlist
total = i+1
sad = 0
notsad = 0
for a,b in zip(song_name, Song_mood):
    st.write(a,':',b)
    if b=='notsad':
        notsad = notsad + 1
    if b=='sad':
        sad = sad + 1

percentage = (sad / i)*100
sadness_prob = sad/i
st.write("Percentage of sadness based on your favourite song preference :",percentage, "%")

stress_level=st.slider("From scale 1 - 10, how stress are you?", 0,10)



if stress_level<=10 and stress_level>=0 :
    stress_prob = stress_level/10
else:
    st.write("Out of range given!")


    


depression_prob = sadness_prob * stress_prob

st.write("Probability of stress:", stress_prob)
st.write("Based on calculation, you have probability of ", depression_prob," to be depressed.")
if depression_prob >= 0.75 :
    st.title("You are having severe depression!")
    st.write("Please contact 1-800-82-0066 or email to info.miasa@gmail.com to recieve help and aid.")
    st.write("Talk more to people.")
elif depression_prob >= 0.35 and depression_prob <0.75 :
    st.title("You are having mild depression.")
    st.write("If you insist, you can contact 1-800-82-0066 or email to info.miasa@gmail.com to get more information about deperssion")
    st.write("Eat healthy food and find good vibes.")
else:
    st.title("Your depression type is normal")
    st.write("Continue having good mental health by doing what you like.")
