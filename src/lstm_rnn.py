import json
import random
from argparse import ArgumentParser

import torch
from sklearn.metrics import accuracy_score, classification_report
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import DataLoader


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.sigmoid(self.fc(hidden.squeeze(0)))


def tokenize_data(data):
    return [(tokenizer(item['text']), item['label']) for item in data]


def yield_tokens(data):
    for text, _ in data:
        yield text


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(vocab(_text), dtype=torch.long)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.long), pad_sequence(text_list, padding_value=0.0)


def evaluate(model, data_loader, criterion, mode="epoch"):
    epoch_loss = 0
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for labels, text in data_loader:
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())

            epoch_loss += loss.item()
            # Round predictions to the closest integer (0 or 1)
            preds = torch.round(predictions)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if mode == "epoch":
        return epoch_loss / len(data_loader), accuracy_score(all_labels, all_preds)
    elif mode == "last":
        return classification_report(all_labels, all_preds)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset_full.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mode", type=str, default="debugging")

    args = parser.parse_args()

    with open(args.dataset_path) as f:
        data_dict = json.load(f)

    tokenizer = get_tokenizer('basic_english')
    specials = ['<unk>', '<pad>']

    print("Tokenizing data...")
    training_data_tokenized = tokenize_data(data_dict['training'])
    validation_data_tokenized = tokenize_data(data_dict['validation'])
    print("Done tokenizing data.")

    # For debugging purposes
    if args.mode == "debugging":
        training_data_tokenized = random.sample(training_data_tokenized, 1000)
        validation_data_tokenized = random.sample(validation_data_tokenized, 400)

    vocab = build_vocab_from_iterator(yield_tokens(training_data_tokenized), specials=specials)
    vocab.set_default_index(vocab["<unk>"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(training_data_tokenized, args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(validation_data_tokenized, batch_size=1, shuffle=False, collate_fn=collate_batch)

    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    print("Start training...")
    accumulation_steps = 4
    for epoch in range(100):
        epoch_loss = 0
        model.train()
        optimizer.zero_grad()
        for i, (labels, text) in enumerate(train_dataloader):
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()
            if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()

        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)
        print(
            f'Epoch: {epoch + 1:02}, '
            f'Train Loss: {epoch_loss:.3f}, '
            f'Val Loss: {valid_loss:.3f}, '
            f'Val Acc: {valid_acc * 100:.2f}%'
        )
    print(evaluate(model, valid_dataloader, criterion, mode="last"))
