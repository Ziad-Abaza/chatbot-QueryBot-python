import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# 1. Load the JSON file
with open('intents.json', 'r') as file:
    data = json.load(file)

# 2. Extract patterns and tags
patterns = []
tags = []
tag_to_patterns = {}  # Map each tag to its patterns
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    tag_to_patterns[intent['tag']] = intent['patterns']

# 3. Encode patterns using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encode tags using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the model
input_layer = Input(shape=(X.shape[1],))
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.3)(x)  # Add Dropout for regularization
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)  # Add Dropout for regularization
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
model.fit(X_train, y_train, epochs=600, batch_size=32, validation_data=(X_test, y_test))

# 6. Generate new data
def generate_new_data(model, vectorizer, label_encoder, tag_to_patterns, num_samples_per_tag=5):
    new_data = {"intents": []}
    int_to_tag = {i: tag for i, tag in enumerate(label_encoder.classes_)}
    tag_to_responses = {intent['tag']: intent['responses'] for intent in data['intents']}
    
    for tag in label_encoder.classes_:
        original_patterns = tag_to_patterns[tag]
        new_patterns = set(original_patterns)  # Avoid duplication
        responses = tag_to_responses[tag]
        
        # Generate new patterns for this tag
        for _ in range(num_samples_per_tag):
            # Try generating a slightly different pattern based on original patterns
            base_pattern = random.choice(original_patterns)
            words = base_pattern.split()
            random.shuffle(words)
            new_pattern = " ".join(words)  # Slightly modified pattern
            
            # Ensure that the new pattern is not already in the set
            if new_pattern not in new_patterns:
                new_patterns.add(new_pattern)
        
        # Select a random response for the tag
        new_intent = {
            "tag": tag,
            "patterns": list(new_patterns),
            "responses": random.sample(responses, min(3, len(responses))),  # Limit to 3 responses
            "context_set": ""
        }
        new_data["intents"].append(new_intent)
    
    return new_data

new_dataset = generate_new_data(model, vectorizer, label_encoder, tag_to_patterns, num_samples_per_tag=15)

# 7. Save new JSON
with open('new_dataset.json', 'w') as file:
    json.dump(new_dataset, file, indent=4)

print("New dataset generated and saved as new_dataset.json")
