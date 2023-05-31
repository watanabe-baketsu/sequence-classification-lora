import json
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer


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
    return {"hidden_state": last_hidden_state[:0].cpu().numpy()}


def visualize_dataset_features(dataset: DatasetDict):
    X_train = np.array(dataset["training"]["hidden_state"])
    X_valid = np.array(dataset["validation"]["hidden_state"])
    y_train = np.array(dataset["training"]["label"])
    y_valid = np.array(dataset["validation"]["label"])
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


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
    training_dataset = dataset["training"].shuffle(seed=42).select(range(500))
    validation_dataset = dataset["validation"].shuffle().select(range(100))
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
    })

    # Tokenize the texts
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(extract_hidden_states, batched=True)

    visualize_dataset_features(dataset)


