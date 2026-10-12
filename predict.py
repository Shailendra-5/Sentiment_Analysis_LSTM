import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("sentiment_lstm.keras")


# Load tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


maxlen = 50


while True:

    sentence = input("\nEnter Sentence: ")

    if sentence.lower() == "exit":
        break


    sequence = tokenizer.texts_to_sequences([sentence])

    padded = pad_sequences(
        sequence,
        maxlen=maxlen
    )


    prediction = model.predict(padded)


    probability = prediction[0][0]


    print("Probability:", probability)


    if probability >= 0.5:
        print("Positive 😊")
    else:
        print("Negative 😡")