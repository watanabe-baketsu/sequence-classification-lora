from argparse import ArgumentParser
import copy

import torch
from datasets import DatasetDict
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from classifiers import SimpleClassifiers, NeuralNetwork
from utils import read_dataset, detailed_report


def encode(data: DatasetDict) -> DatasetDict:
    """
    :param data:
    :return:
    """
    # encode the texts
    encoding = processor(data['text'], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
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


def evaluate(nn_model: NeuralNetwork, mode: str = "epoch"):
    # バリデーションフェーズ
    nn_model.eval()
    all_labels = []
    all_predictions = []
    if mode == "epoch":
        valid_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = nn_model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                # ラベルと予測結果を保存
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend((outputs > 0.5).cpu().numpy())

        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        return epoch_accuracy
    # メトリクスを計算
    elif mode == "last":
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = nn_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # ラベルと予測結果を保存
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend((outputs > 0.5).cpu().numpy())

        print(f'Testing Loss: {test_loss / len(test_loader)}')
        print("#### NeuralNetwork")
        print(classification_report(all_labels, all_predictions))
        detailed_report(all_labels, all_predictions)


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
    dataset = DatasetDict({
        "training": training_dataset,
        "validation": validation_dataset,
        "testing": testing_dataset
    })

    # Encode the texts
    dataset = dataset.map(encode, batched=True, batch_size=50)
    # Convert some columns to torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Extract hidden states
    dataset = dataset.map(extract_hidden_states, batched=True, batch_size=args.gpu_batch_size)

    # Train simple classifiers and evaluate them
    classifiers = SimpleClassifiers(dataset)
    classifiers.evaluate_all()

    # Train neural network and evaluate it
    train_hidden_states = dataset["training"]["hidden_state"]
    train_label = torch.tensor(dataset["training"]["label"]).float().view(-1, 1)
    valid_hidden_states = dataset["validation"]["hidden_state"]
    valid_label = torch.tensor(dataset["validation"]["label"]).float().view(-1, 1)
    test_hidden_states = dataset["testing"]["hidden_state"]
    test_label = torch.tensor(dataset["testing"]["label"]).float().view(-1, 1)

    train_dataset = TensorDataset(train_hidden_states, train_label)
    valid_dataset = TensorDataset(valid_hidden_states, valid_label)
    test_dataset = TensorDataset(test_hidden_states, test_label)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_size = len(train_hidden_states[0])
    nn_model = NeuralNetwork(input_size).to(args.device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(nn_model.parameters())
    schedular = CosineAnnealingLR(optimizer, T_max=100)

    best_model = copy.deepcopy(nn_model)
    best_accuracy = 0.0
    epochs = 100
    for epoch in range(epochs):
        # トレーニングフェーズ
        nn_model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()

            outputs = nn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # パラメータの更新
            running_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, Training Loss: {running_loss / len(train_loader)}')
        tmp_accuracy = evaluate(nn_model, mode="epoch")
        if tmp_accuracy > best_accuracy:
            best_accuracy = tmp_accuracy
            print(f"Best accuracy: {best_accuracy}")
            best_model = copy.deepcopy(nn_model)
        schedular.step()

    print('Finished Training')
    evaluate(best_model, mode="last")

