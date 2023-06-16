import json
from argparse import ArgumentParser

import torch
from datasets import Dataset, DatasetDict
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor, Trainer

from utils_and_classifiers import create_report


def read_validation_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "testing": Dataset.from_list(data["testing"]),
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="../tuned_models/deberta-base-mnli")
    parser.add_argument("--validation_dataset", type=str, default="../dataset/dataset.json")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "markuplm"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    config = PeftConfig.from_pretrained(args.model_name)

    if args.mode == "markuplm":
        tokenizer = AutoProcessor.from_pretrained(config.base_model_name_or_path)
    elif args.mode == "default":
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        raise ValueError("Invalid mode")

    id2label = {0: "not-phish", 1: "phish"}
    label2id = {"not-phish": 0, "phish": 1}

    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        local_files_only=False,
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(inference_model, args.model_name).to(args.device)

    validation_dataset = read_validation_dataset(args.validation_dataset)
    tokenized_validation_dataset = validation_dataset.map(tokenize, batched=True, remove_columns=["text"])

    trained_model = Trainer(model=model)

    # Evaluate the model
    create_report(tokenized_validation_dataset, trained_model)
