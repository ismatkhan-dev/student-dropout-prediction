import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, sep=";")
    return data


def preprocess_data(data):
    X = data.drop("Target", axis=1)
    y = data["Target"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


if __name__ == "__main__":

    data = load_data("data.csv")

    X, y = preprocess_data(data)

    print("Data preprocessing completed!")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)