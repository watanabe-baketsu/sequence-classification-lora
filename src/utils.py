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
        "testing": Dataset.from_list(data["testing"])
    })

    return dataset


def create_report(dataset: DatasetDict, trainer: Trainer):
    # Get the predictions
    preds_output = trainer.predict(dataset["testing"])
    predictions = np.argmax(preds_output.predictions, axis=1)

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    print(accuracy.compute(predictions=predictions, references=dataset["testing"]["label"]))
    print(recall.compute(predictions=predictions, references=dataset["testing"]["label"]))
    print(precision.compute(predictions=predictions, references=dataset["testing"]["label"]))
    print(f1.compute(predictions=predictions, references=dataset["testing"]["label"]))

    print(">> Classification Report <<")
    print(classification_report(dataset["testing"]["label"], predictions, target_names=["not-phish", "phish"]))


def detailed_report(labels, preds):
    # Load the evaluation metrics
    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    print(accuracy.compute(predictions=preds, references=labels))
    print(recall.compute(predictions=preds, references=labels))
    print(precision.compute(predictions=preds, references=labels))
    print(f1.compute(predictions=preds, references=labels))
