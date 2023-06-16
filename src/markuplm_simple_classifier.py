from argparse import ArgumentParser

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoProcessor

from utils_and_classifiers import SimpleClassifiers, visualize_dataset_features, read_dataset


def encode(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # encode the texts
    encoding = processor(data['text'], padding=True, truncation=True, return_tensors="pt")
    return encoding


def extract_hidden_states(batch):
    """
    :param batch:
    :return:
    """
    inputs = {k: v.to(args.device) for k, v in batch.items() if k in processor.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/markuplm-base")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--gpu_batch_size", type=int, default=30)

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)

    # Load dataset
    dataset = read_dataset(args.dataset_path)
    training_dataset = dataset["training"].shuffle(seed=42)
    validation_dataset = dataset["validation"].shuffle()
    testing_dataset = dataset["testing"].shuffle()
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    print(f"Testing dataset count: {len(testing_dataset)}")
    valid_test_dataset = Dataset.from_list(
        Dataset.to_list(validation_dataset) + Dataset.to_list(testing_dataset)
    )
    print(f"Validation + Testing dataset count: {len(valid_test_dataset)}")
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": valid_test_dataset,
    })

    # Encode the texts
    dataset = dataset.map(encode, batched=True, batch_size=50)
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Visualize the dataset features
    visualize_dataset_features(dataset)

    # Train simple classifiers and evaluate them
    classifiers = SimpleClassifiers(dataset)
    classifiers.evaluate_all()
