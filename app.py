from flask import Flask, render_template, request

import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)


# Load model
model = load_model("sentiment_lstm.keras")


# Load tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


maxlen = 50


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None
    sentence = ""


    if request.method == "POST":

        sentence = request.form["sentence"]


        sequence = tokenizer.texts_to_sequences([sentence])


        padded = pad_sequences(
            sequence,
            maxlen=maxlen
        )


        prediction = model.predict(padded)


        probability = float(prediction[0][0])


        if probability >= 0.5:
            result = "Positive 😊"
        else:
            result = "Negative 😡"


        probability = round(probability * 100, 2)


    return render_template(
        "index.html",
        result=result,
        probability=probability,
        sentence=sentence
    )


if __name__ == "__main__":
    app.run(debug=True)