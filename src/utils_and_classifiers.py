import json

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from transformers import Trainer
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
        clf.fit(self.X_valid, self.y_valid)
        y_pred = clf.predict(self.X_valid)
        print("#### Dummy Classifier Report")
        print(classification_report(self.y_valid, y_pred))

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, max_iter=3000)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Logistic Regression Report")
        print(classification_report(self.y_valid, y_pred))

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Random Forest Report")
        print(classification_report(self.y_valid, y_pred))

    def xgboost(self):
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### XGBoost Report")
        print(classification_report(self.y_valid, y_pred))

    def support_vector_machine(self):
        from sklearn.svm import SVC

        clf = SVC(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Support Vector Machine Report")
        print(classification_report(self.y_valid, y_pred))

    def k_nearest_neighbors(self):
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### K Nearest Neighbors Report")
        print(classification_report(self.y_valid, y_pred))

    def neural_network(self):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(random_state=0, max_iter=3000)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Neural Network Report")
        print(classification_report(self.y_valid, y_pred))

    def evaluate_all(self):
        self.dummy_classifier()
        self.logistic_regression()
        self.random_forest()
        self.xgboost()
        self.support_vector_machine()
        self.k_nearest_neighbors()
        self.neural_network()


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


def create_report(dataset: DatasetDict, trainer: Trainer):
    from sklearn.metrics import classification_report

    # Get the predictions
    preds_output = trainer.predict(dataset["validation"])
    predictions = np.argmax(preds_output.predictions, axis=1)

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    print(accuracy.compute(predictions=predictions, references=dataset["validation"]["label"]))
    print(recall.compute(predictions=predictions, references=dataset["validation"]["label"]))
    print(precision.compute(predictions=predictions, references=dataset["validation"]["label"]))
    print(f1.compute(predictions=predictions, references=dataset["validation"]["label"]))

    print(">> Classification Report <<")
    print(classification_report(dataset["validation"]["label"], predictions, target_names=["not-phish", "phish"]))