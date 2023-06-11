from argparse import ArgumentParser
from typing import Dict

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np

from utils_and_classifiers import read_dataset


def encode(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # encode the texts
    encoding = processor(data['text'], padding=True, truncation=True, return_tensors="pt")
    return encoding


def tokenize(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # Tokenize the texts
    tokenized_inputs = tokenizer(data['text'], padding=True, truncation=True, return_tensors="pt")
    return tokenized_inputs


def compute_metrics(p) -> Dict[str, float]:
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)

    return accuracy.compute(predictions=predictions, references=labels)


def create_report(dataset: DatasetDict, trainer: Trainer):
    from sklearn.metrics import classification_report

    # Get the predictions
    preds_output = trainer.predict(dataset["validation"])
    predictions = np.argmax(preds_output.predictions, axis=1)

    print(">> Classification Report <<")
    print(classification_report(dataset["validation"]["label"], predictions))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base-mnli",
                        help="Name of the model to be used for training")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json", help="Training dataset")
    parser.add_argument("--target_modules", nargs='+', default=["in_proj"])
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--mode", type=str, default="default", choices=["default", "markuplm"])

    args = parser.parse_args()

    # Load dataset
    htmls = read_dataset(args.dataset_path)
    training_dataset = htmls["training"].shuffle()
    validation_dataset = htmls["validation"].shuffle()
    print(f"Training dataset count: {len(training_dataset)}")
    print(f"Validation dataset count: {len(validation_dataset)}")
    htmls = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.mode == "markuplm":
        processor = AutoProcessor.from_pretrained(args.model_name)
        tokenized_dataset = htmls.map(encode, batched=True, remove_columns=["text"])
    elif args.mode == "default":
        tokenized_dataset = htmls.map(tokenize, batched=True, remove_columns=["text"])
    else:
        raise ValueError("Invalid mode")

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
        r=args.lora_r,  # 8
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
        target_modules=args.target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    batch_size = args.batch_size  # 4
    training_args = TrainingArguments(
        output_dir=f"../tuned_models/{args.model_name.split('/')[-1]}",  # output directory
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=3000,
        save_strategy="steps",
        lr_scheduler_type="cosine",
    )

    accuracy = evaluate.load("accuracy")
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

    if "/" in args.model_name:
        save_directory = f"../tuned_models/{args.model_name.split('/')[-1]}"
    else:
        save_directory = f"../tuned_models/{args.model_name}"
    trainer.model.save_pretrained(save_directory=save_directory)

    # create classification report
    create_report(tokenized_dataset, trainer)
