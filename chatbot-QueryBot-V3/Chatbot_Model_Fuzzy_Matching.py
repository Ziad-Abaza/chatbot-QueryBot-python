import nltk
nltk.download('punkt')  # Download punkt tokenizer for word tokenization
from nltk.stem.lancaster import LancasterStemmer  # Importing Lancaster Stemmer for stemming
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import json
import pickle
import os
from fuzzywuzzy import fuzz  # For finding fuzzy matches between strings
from time import sleep

# Read the intents.json file which contains all intents and their associated patterns and responses
with open("dataset.json") as file:
    data = json.load(file)

# Try to load preprocessed data if available, otherwise process the raw data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)  # Load existing data
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Process each intent and its patterns to create the dataset
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # Tokenize the pattern into words
            words.extend(wrds)  # Add the words to the list of all words
            docs_x.append(wrds)  # Add the words to the list of input patterns
            docs_y.append(intent["tag"])  # Add the associated intent tag

        if intent["tag"] not in labels:  # Ensure the tag is unique
            labels.append(intent["tag"])

    # Perform stemming and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # Stem each word and ignore question marks
    words = sorted(list(set(words)))  # Remove duplicates and sort the words
    labels = sorted(labels)  # Sort the labels

    training = []
    output = []

    # Initialize an empty output array for each possible response
    out_empty = [0 for _ in range(len(labels))]

    # Create the bag of words for each pattern
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]  # Stem the words in the document

        for w in words:  # Create a bag of words (1 if word is present, 0 if absent)
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Create output row for the current pattern based on the tag
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1  # Set the corresponding label

        training.append(bag)  # Add the bag to the training data
        output.append(output_row)  # Add the output to the output data

    # Convert the training and output data to numpy arrays
    training = np.array(training)
    output = np.array(output)

    # Save the processed data to avoid reprocessing in the future
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Build the neural network model using Keras
model = Sequential([
    Dense(128, input_shape=(len(training[0]),), activation='relu'),  # Input layer with 128 nodes
    Dense(64, activation='relu'),  # Hidden layer with 64 nodes
    Dense(len(output[0]), activation='softmax')  # Output layer with the number of possible labels
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check if a pre-trained model exists and load it, otherwise train a new model
if os.path.exists("model.h5"):
    model.load_weights("model.h5")  # Load model weights if available
    print("Model loaded successfully!")
else:
    print("Training the model...")
    model.fit(training, output, epochs=400, batch_size=32, verbose=1)  # Train the model
    model.save("model.h5")  # Save the trained model
    print("Model trained and saved!")

# Function to convert a sentence into a bag of words representation
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  # Initialize a bag with zeros

    s_words = nltk.word_tokenize(s)  # Tokenize the input sentence
    s_words = [stemmer.stem(word.lower()) for word in s_words]  # Stem the words

    for se in s_words:  # For each stemmed word in the input sentence
        for i, w in enumerate(words):
            if w == se:  # If the word exists in the known words
                bag[i] = 1  # Mark the word as present

    return np.array(bag)

# Function to find a similar question using fuzzy matching
def find_similar_question(command, intents_data, threshold=70):
    best_match = 0
    best_response = None

    # Loop through each intent and its patterns
    for intent in intents_data:
        for pattern in intent['patterns']:
            similarity = fuzz.ratio(command.lower(), pattern.lower())  # Compute similarity between input and pattern
            if similarity > best_match:
                best_match = similarity
                best_response = random.choice(intent['responses'])  # Choose a random response for the best match
    
    if best_match >= threshold:  # If similarity exceeds the threshold
        return best_response
    return None  # Return None if no sufficient match is found

# Function to start the chatbot
def chat():
    print("How can I help you?")
    while True:
        inp = input("You: ")  # Get user input
        if inp.lower() == "quit":  # If the user types "quit", exit the loop
            break

        # Convert the input into a bag of words representation
        results = model.predict(np.array([bag_of_words(inp, words)]))[0]
        results_index = np.argmax(results)  # Get the index of the highest predicted value
        tag = labels[results_index]  # Get the tag corresponding to the highest prediction

        if results[results_index] > 0.9:  # If the prediction is highly confident
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']  # Get the responses for the predicted tag
            sleep(3)  # Simulate thinking delay
            Bot = random.choice(responses)  # Choose a random response
            print("Answer:", Bot)
        else:
            # Search for the closest match using fuzzy matching
            similar_response = find_similar_question(inp, data["intents"])
            if similar_response:
                print("Answer (Closest Match):", similar_response)
            else:
                print("I don't understand!")  # If no match is found, respond with "I don't understand!"

chat()  # Start the chatbot
