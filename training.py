#This training file is created based on the following tutorial: https://www.youtube.com/watch?v=1lwddP0KUEg
import random
import json
import pickle
import numpy as np
import nltk

from HanTa import HanoverTagger as ht
hannover = ht.HanoverTagger('morphmodel_ger.pgz')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

intents = json.loads(open("intents.json").read()) #reading the content of the json file

words = [] #empty list for words we are going to have
classes = [] #empty list for classes we are going to have
documents = [] #empty list for what is going to be a combination of "belongings"
ignore_letters = ["?", "!", ".", ","] #list of characters which are not important for learning

for intent in intents["intents"]: #we iterate over every dictionary in the intents file
    for pattern in intent["patterns"]: #for each pattern in dictionary
        word_list = nltk.word_tokenize(pattern) #split each pattern into individual words
        words.extend(word_list) #add the collection of individual words to the word list
        documents.append((word_list, intent["tag"])) #add a word list and a respective tag to the document
        if intent["tag"] not in classes: #check if a tag already exists in the list of classes
            classes.append(intent["tag"]) #add it if it doesn't

words = [hannover.analyze(word)[0] for word in words if word not in ignore_letters] #lemmatize words
words = sorted(set(words)) #eliminate duplicates in words
classes = sorted(set(classes)) #eliminate duplicates in classes

pickle.dump(words, open("words.pkl", "wb")) #save words into a pickle file
pickle.dump(classes, open("classes.pkl", "wb")) #save classes into a pickle file

training = []
output_empty = [0] * len(classes)

for document in documents: #for each document
    bag = [] #create an empty bag of words
    word_patterns = document[0]
    word_patterns = [hannover.analyze(word.lower())[0] for word in word_patterns] #lemmatize word patterns
    for word in words: #check if a particular word from words occurs in a pattern
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row]) #append the training list 

random.shuffle((training)) #shuffle the data
training = np.array(training) #turn the data into a numpy array

train_x = list(training[:, 0]) #features
train_y = list(training[:, 1]) #labels

model = Sequential() #build the neural network
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu")) #1st dense layer
model.add(Dropout(0.5)) #a dropout layer to prevent overfitting
model.add(Dense(64, activation="relu")) #another dense layer
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax")) 

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Stochastic Gradient Descent optimizer
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) #compile all layers

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) #save model with intents data into a variable
model.save("chatbot_model.h5", hist) #save model with the return value of a fit function into an h5 file
print("Done")
