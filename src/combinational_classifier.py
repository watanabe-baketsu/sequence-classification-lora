from argparse import ArgumentParser

import numpy as np
from datasets import DatasetDict
from sklearn.metrics import classification_report

from classifiers import TransformerBody, MarkupLMBody, NNTrainerUtility
from utils import read_dataset


def nn_head(text_preds: list, markup_preds: list, labels: list):
    from sklearn.neural_network import MLPClassifier
    inputs = [[t[0], m[0]] for t, m in zip(text_preds, markup_preds)]
    x = np.array(inputs)
    y = np.array(labels)
    clf = MLPClassifier(random_state=0, max_iter=1000, early_stopping=True)
    clf.fit(x, y)
    return clf


def xgb_head(text_preds: list, markup_preds:list, labels: list):
    from xgboost import XGBClassifier
    inputs = [[t[0], m[0]] for t, m in zip(text_preds, markup_preds)]
    x = np.array(inputs)
    y = np.array(labels)
    clf = XGBClassifier(random_state=0, max_iter=1000)
    clf.fit(x, y)
    return clf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text_model", type=str, default="microsoft/deberta-base-mnli")
    parser.add_argument("--markup_model", type=str, default="microsoft/markuplm-base")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full_collection.json")
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

    text_model = TransformerBody(args.text_model, args.device)
    markup_model = MarkupLMBody(args.markup_model, args.device)

    # Preparation of visible text classifier
    text_dataset = dataset.map(text_model.tokenize, batched=True, batch_size=50)
    text_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    text_dataset = text_dataset.map(text_model.extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Preparation of markup classifier
    markup_dataset = dataset.map(markup_model.encode, batched=True, batch_size=50)
    markup_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    markup_dataset = markup_dataset.map(markup_model.extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Train neural network and evaluate it
    trainer = NNTrainerUtility(args.device)
    print("Training text classifier")
    text_nn_model = trainer.train_nn_model(text_dataset)
    print("Training markup classifier")
    markup_nn_model = trainer.train_nn_model(markup_dataset)

    # Train last neural network
    text_outputs_train = trainer.extract_outputs(text_nn_model, text_dataset, "training")
    markup_outputs_train = trainer.extract_outputs(markup_nn_model, markup_dataset, "training")
    last_nn_model = nn_head(text_outputs_train, markup_outputs_train, dataset["training"]["label"])
    last_xgb_model = xgb_head(text_outputs_train, markup_outputs_train, dataset["training"]["label"])

    # Test the combination of classifiers
    text_outputs_test = trainer.extract_outputs(text_nn_model, text_dataset, "testing")
    markup_outputs_test = trainer.extract_outputs(markup_nn_model, markup_dataset, "testing")
    inputs = [[t[0], m[0]] for t, m in zip(text_outputs_test, markup_outputs_test)]
    x = np.array(inputs)
    final_outputs = last_nn_model.predict(x)

    # Calculate the metrics NN
    labels = dataset["testing"]["label"]
    print("Combination Classification Report(NN)")
    print(classification_report(labels, final_outputs, digits=6))

    # Calculate the metrics XGB
    final_outputs = last_xgb_model.predict(x)
    print("Combination Classification Report(XGB)")
    print(classification_report(labels, final_outputs, digits=6))
