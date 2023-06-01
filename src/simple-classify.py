import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer
from matplotlib import pyplot as plt
from umap import UMAP


class SimpleClassifiers:
    def __init__(self, dataset: DatasetDict):
        self.X_train = np.array(dataset["training"]["hidden_state"])
        self.y_train = np.array(dataset["training"]["label"])
        self.X_valid = np.array(dataset["validation"]["hidden_state"])
        self.y_valid = np.array(dataset["validation"]["label"])

    def dummy_classifier(self):  # baseline
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> Dummy Classifier Report <<")
        print(classification_report(self.y_valid, y_pred))

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, max_iter=3000)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> Logistic Regression Report <<")
        print(classification_report(self.y_valid, y_pred))

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> Random Forest Classifier Report <<")
        print(classification_report(self.y_valid, y_pred))

    def support_vector_machine(self):
        from sklearn.svm import SVC

        clf = SVC(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> Support Vector Machine Report <<")
        print(classification_report(self.y_valid, y_pred))

    def k_nearest_neighbors(self):
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> K Nearest Neighbors Report <<")
        print(classification_report(self.y_valid, y_pred))

    def newral_network(self):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(random_state=0, max_iter=3000)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print(">> Neural Network Report <<")
        print(classification_report(self.y_valid, y_pred))

def read_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "training": Dataset.from_list(data["training"]),
        "validation": Dataset.from_list(data["validation"]),
    })

    return dataset


def tokenize(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # Tokenize the texts
    tokenized_inputs = tokenizer(data['text'], padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs


def extract_hidden_states(batch):
    """
    :param batch:
    :return:
    """
    inputs = {k: v.to(args.device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


def visualize_dataset_features(dataset: DatasetDict):
    X_train = np.array(dataset["training"]["hidden_state"])
    y_train = np.array(dataset["training"]["label"])

    # Scale the features
    X_scaled = MinMaxScaler().fit_transform(X_train)
    # initialize umap and fit the scaled features
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    # create the embeddings
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train
    df_emb.head()

    fig, axes = plt.subplots(2, 1, figsize=(5, 5))
    axes = axes.flatten()
    cmaps = ["Blues", "Reds"]
    labels = ["not-phish", "phish"]

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)

    # Load dataset
    dataset = read_dataset(args.dataset_path)
    training_dataset = dataset["training"].shuffle(seed=42)
    validation_dataset = dataset["validation"].shuffle()
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
    })

    # Tokenize the texts
    dataset = dataset.map(tokenize, batched=True, batch_size=50)
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(extract_hidden_states, batched=True, batch_size=30)

    # Visualize the dataset features
    visualize_dataset_features(dataset)

    # Train simple classifiers and evaluate them
    classifiers = SimpleClassifiers(dataset)
    classifiers.dummy_classifier()
    classifiers.logistic_regression()
    classifiers.random_forest()
    classifiers.support_vector_machine()
    classifiers.k_nearest_neighbors()
    classifiers.newral_network()
