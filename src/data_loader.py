
import pandas as pd
import os

def load_isot_dataset(data_dir="data/isot_dataset"):
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df["text"] = df["title"] + " " + df["text"]

    return df


def load_welfake_dataset(data_dir="data/welfake_dataset"):
    df = pd.read_csv(
        os.path.join(data_dir, "WELFake_Dataset.csv")
    )

    df = df.dropna(subset=["title", "text"])
    df["text"] = df["title"] + " " + df["text"]

    return df


def split_dataset(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
