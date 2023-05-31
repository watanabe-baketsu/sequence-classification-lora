import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
from matplotlib import pyplot as plt
from umap import UMAP


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

    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
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
    training_dataset = dataset["training"].shuffle(seed=42).select(range(200))
    validation_dataset = dataset["validation"].shuffle().select(range(50))
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
    })

    # Tokenize the texts
    dataset = dataset.map(tokenize, batched=True, batch_size=None)
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(extract_hidden_states, batched=True)

    # Visualize the dataset features
    visualize_dataset_features(dataset)


