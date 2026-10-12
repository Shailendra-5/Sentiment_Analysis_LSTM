import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("train.csv", encoding="utf-8", on_bad_lines="skip")

print("Dataset Loaded")
print(data.head())
print(data.shape)


# ==========================
# Clean Dataset
# ==========================

# Remove empty rows
data = data.dropna()

# Convert label safely
data["label"] = (
    data["label"]
    .astype(str)
    .str.replace('"', '')
    .str.strip()
)

data["label"] = data["label"].astype(int)


texts = data["text"].astype(str).values
labels = data["label"].values


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)


# ==========================
# Tokenization
# ==========================

max_words = 5000
maxlen = 50


tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=maxlen
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=maxlen
)


# Save tokenizer

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)


# ==========================
# LSTM Model
# ==========================

model = Sequential()

model.add(
    Embedding(
        max_words,
        128,
        input_length=maxlen
    )
)

model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==========================
# Training
# ==========================

history = model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(
        X_test_pad,
        y_test
    )
)


# ==========================
# Evaluation
# ==========================

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test
)

print("Test Accuracy:", accuracy)


# ==========================
# Save Model
# ==========================

model.save("sentiment_lstm.keras")

print("Model saved successfully")