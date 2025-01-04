# Chatbot-QueryBot-Python

This repository contains two versions of a chatbot built using Natural Language Processing (NLP) techniques. The chatbot can interact with users by processing input and predicting appropriate responses based on pre-defined intents. The models use different machine learning frameworks: **tflearn** and **Keras**.

## Versions:

### Version 1: `chatbot-QueryBot-V1`
- **Framework:** `tflearn`
- **Model Type:** Neural Network
- **Training Data:** Uses intents data stored in the `intents.json` file.

#### Files:
- `main.py`: The main script to train and run the chatbot.
  - Downloads necessary NLP packages (`nltk`).
  - Preprocesses the input text and prepares the training data.
  - Uses `tflearn` to build a neural network with two hidden layers.
  - Saves the trained model to `model.tflearn` file.

### Version 2: `chatbot-QueryBot-V2`
- **Framework:** `Keras` and `tflearn`
- **Model Type:** Neural Network (with both Keras and tflearn implementations)
- **Training Data:** Uses intents data stored in the `intents.json` file.

#### Files:
- `ChatBot_NLP_Model_keras.py`: A Keras-based version of the chatbot.
  - Similar structure to Version 1, but uses Keras layers to build the model.
  - Trains the model for a larger number of epochs and saves it as `model.h5`.
  
- `ChatBot_NLP_Model_tflearn.py`: An alternative `tflearn` implementation.
  - Uses `tflearn` to create a model with multiple hidden layers.
  - Trains the model and saves it as `model.tflearn`.

## Requirements:

- Python 3.x
- Install necessary libraries:
  - `tensorflow`
  - `tflearn`
  - `nltk`
  - `numpy`
  - `json`
  - `pickle`

You can install the required dependencies with the following:
```bash
pip install tensorflow tflearn nltk numpy
```

## Running the Chatbot:

1. Download or clone the repository.
2. Navigate to the folder containing the project files.
3. Run either version:

- For Version 1:
```bash
python chatbot-QueryBot-V1/main.py
```

- For Version 2:
```bash
python chatbot-QueryBot-V2/ChatBot_NLP_Model_keras.py
```

Or if you prefer the `tflearn` version in Version 2:
```bash
python chatbot-QueryBot-V2/ChatBot_NLP_Model_tflearn.py
```

4. Enter text and interact with the bot. Type `"quit"` to exit the conversation.

## Model Training:

- When you run the chatbot for the first time, it will train the model using the data from `intents.json` and save it as `data.pickle`.
- If the model is pre-trained and the pickle file exists, it will directly load the model data.

## Usage:

- The chatbot responds to various patterns defined in the `intents.json` file. Each pattern is associated with a tag, and the chatbot predicts the most likely response based on the user’s input.

## Intents Format:

The `intents.json` file defines different patterns and their corresponding responses. An example format:

```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you?"],
            "responses": ["Hello!", "Hi, how can I help?", "Greetings!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Take care!"]
        }
    ]
}
```

## Model Architecture:

- **Input Layer:** Accepts a sequence of words tokenized from the user input.
- **Hidden Layers:** Two or more fully connected layers with ReLU activation.
- **Output Layer:** A softmax layer that outputs probabilities for each intent.

## Future Improvements:
- Adding more advanced NLP models.
- Integrating with external APIs for dynamic responses.
- Expanding the intents dataset for a broader range of interactions.

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
