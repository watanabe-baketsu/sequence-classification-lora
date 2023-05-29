import json
from argparse import ArgumentParser

import torch
from datasets import Dataset, DatasetDict
from peft import get_peft_config, PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_validation_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "validation": Dataset.from_list(data["validation"]),
    })

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="../tuned_models/deberta-base-mnli")
    parser.add_argument("--validation_dataset", type=str, default="../dataset/dataset.json")

    args = parser.parse_args()
    config = PeftConfig.from_pretrained(args.model_name)

    id2label = {0: "not-phish", 1: "phish"}
    label2id = {"not-phish": 0, "phish": 1}

    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(inference_model, args.model_name)

    validation_dataset = read_validation_dataset(args.validation_dataset)
    validation_dataset = validation_dataset["validation"].shuffle().select(range(5))

    for data in validation_dataset:
        inputs = tokenizer(data["text"], padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        print(f"Predicted class: {id2label[predicted_class_id]} / Actual class: {data['label']}")
