from argparse import ArgumentParser
from typing import Dict
import json

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np


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


def preprocess_function(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # Tokenize the texts
    tokenized_inputs = tokenizer(data['text'], padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs


def compute_metrics(p) -> Dict[str, float]:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    return {
        "precision": precision.compute(predictions=predictions, references=labels)["precision"],
        "recall": recall.compute(predictions=predictions, references=labels)["recall"],
        "f1": f1.compute(predictions=predictions, references=labels)["f1"],
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli",
                        help="Name of the model to be used for training")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json", help="Training dataset")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    htmls = read_dataset(args.dataset_path)
    training_dataset = htmls["training"].shuffle(seed=42).select(range(3000))
    validation_dataset = htmls["validation"].shuffle().select(range(500))
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    htmls = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
    })
    tokenized_dataset = htmls.map(preprocess_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "not-phish", 1: "phish"}
    label2id = {"not-phish": 0, "phish": 1}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=4,  # 8
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
        target_modules=["in_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    batch_size = 4
    training_args = TrainingArguments(
        output_dir=f"../tuned_models/{args.model_name.split('/')[-1]}",  # output directory
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["training"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_directory = f"../tuned_models/{args.model_name.split('/')[-1]}"
    trainer.model.save_pretrained(save_directory=save_directory)
