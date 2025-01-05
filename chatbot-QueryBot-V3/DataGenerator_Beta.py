import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load and verify JSON file
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        return json.load(file)

data = load_json('intents.json')

# 2. Extract patterns and tags
def extract_data(data):
    patterns, tags, tag_to_patterns = [], [], {}
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
        tag_to_patterns[intent['tag']] = intent['patterns']
    return patterns, tags, tag_to_patterns

patterns, tags, tag_to_patterns = extract_data(data)

# 3. Encode patterns and tags
def preprocess_data(patterns, tags):
    vectorizer = TfidfVectorizer(max_features=500)  # Limit features for efficiency
    X = vectorizer.fit_transform(patterns).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tags)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer, label_encoder

X_train, X_test, y_train, y_test, vectorizer, label_encoder = preprocess_data(patterns, tags)

# 4. Build model
def build_model(input_dim, num_classes):
    input_layer = Input(shape=(input_dim,))
    x = Dense(256)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

model = build_model(X_train.shape[1], len(label_encoder.classes_))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model with callbacks
def train_model(model, X_train, y_train, X_test, y_test, batch_size=16, epochs=200):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

history = train_model(model, X_train, y_train, X_test, y_test)

# 6. Evaluate model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss on test data: {loss * 100:.2f}%")
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    
evaluate_model(model, X_test, y_test)

# 7. Generate new dataset
def generate_new_data(tag_to_patterns, num_samples_per_tag=5):
    new_data = {"intents": []}
    for tag, original_patterns in tag_to_patterns.items():
        new_patterns = set(original_patterns)
        for _ in range(num_samples_per_tag):
            base_pattern = random.choice(original_patterns)
            words = base_pattern.split()
            random.shuffle(words)
            new_pattern = " ".join(words)
            new_patterns.add(new_pattern)

        new_data["intents"].append({
            "tag": tag,
            "patterns": list(new_patterns),
            "responses": random.sample(tag_to_patterns[tag], min(3, len(tag_to_patterns[tag]))),
            "context_set": ""
        })
    return new_data

new_dataset = generate_new_data(tag_to_patterns, num_samples_per_tag=15)

# 8. Save new dataset
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

save_json(new_dataset, 'dataset.json')
print("New dataset saved as 'dataset.json'.")
