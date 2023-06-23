from argparse import ArgumentParser

import torch
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer

from utils_and_classifiers import SimpleClassifiers, read_dataset, visualize_dataset_features


def tokenize(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # Tokenize the texts
    tokenized_inputs = tokenizer(data['text'], padding="max_length", max_length=50, truncation=True, return_tensors="pt")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--gpu_batch_size", type=int, default=30)

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
    dataset = dataset.map(extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Visualize the dataset features
    visualize_dataset_features(dataset)

    # Train simple classifiers and evaluate them
    classifiers = SimpleClassifiers(dataset)
    classifiers.evaluate_all()
