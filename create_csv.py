import pandas as pd


data = [
    ["The acting was brilliant, the story was amazing", 1],
    ["This movie was boring, I hated it", 0],
    ["I loved this film", 1],
    ["The story was terrible", 0],
    ["The actors performed amazingly", 1]
]


df = pd.DataFrame(
    data,
    columns=["text", "label"]
)


df.to_csv(
    "dataset/train.csv",
    index=False
)


print("CSV created successfully")