import pandas as pd

df = pd.read_csv(
    "train.csv",
    quotechar='"',
    on_bad_lines="skip"
)

df.to_csv(
    "clean_train.csv",
    index=False
)

print("Dataset cleaned successfully ✅")
print(df.shape)