from argparse import ArgumentParser

from datasets import DatasetDict

from classifiers import SimpleClassifiers, TransformerBody, NNTrainerUtility
from utils import read_dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full_collection_with_visible_text.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--gpu_batch_size", type=int, default=30)

    args = parser.parse_args()

    # Load dataset
    dataset = read_dataset(args.dataset_path)
    training_dataset = dataset["training"].shuffle(seed=42)
    validation_dataset = dataset["validation"].shuffle()
    testing_dataset = dataset["testing"].shuffle()
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
        "testing": testing_dataset
    })

    transformer = TransformerBody(args.model_name, args.device)

    # Tokenize the texts
    dataset = dataset.map(transformer.tokenize, batched=True, batch_size=50)
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(transformer.extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Train simple classifiers and evaluate them
    classifiers = SimpleClassifiers(dataset)
    classifiers.evaluate_all()

    # Train neural network and evaluate it
    trainer = NNTrainerUtility(args.device)
    trainer.train_nn_model(dataset)
