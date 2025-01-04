import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
import tflearn
import random
import json
import pickle

from time import sleep

# Reading the intents.json file, which contains chatbot conversation data
with open("intents.json") as file:
    data = json.load(file)

# Checking if the model's preprocessed data already exists
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Extracting words and tags from the conversation data
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Preparing words and tags for training
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    # Saving the preprocessed model data for future use
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Building the neural network model using tflearn
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 32)  # First layer with 32 units
net = tflearn.fully_connected(net, 32)  # Second layer with 32 units
net = tflearn.fully_connected(net, 32)  # Third layer with 32 units
net = tflearn.fully_connected(net, 32)  # Fourth layer with 32 units
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # Output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Uncomment the next lines if you want to load a pre-trained model
# try:
#     model.load("model.tflearn")
# except:

# Training the model with the available data
model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# Function to convert user input into a numerical representation (bag of words)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# Function to start a chat session
def chat():
    print("How can I help you?")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Predicting the user's intent
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # Verifying prediction confidence and responding accordingly
        if results[results_index] > 0.9:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print("Answer:", Bot)
        else:
            print("I don't understand!")

chat()
