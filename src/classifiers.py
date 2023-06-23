import copy

import numpy as np
import torch
from torch import nn
from datasets import DatasetDict
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import detailed_report


class SimpleClassifiers:
    def __init__(self, dataset: DatasetDict):
        self.X_train = np.array(dataset["training"]["hidden_state"])
        self.y_train = np.array(dataset["training"]["label"])
        self.X_valid = np.array(dataset["validation"]["hidden_state"])
        self.y_valid = np.array(dataset["validation"]["label"])

    def dummy_classifier(self):  # baseline
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(self.X_valid, self.y_valid)
        y_pred = clf.predict(self.X_valid)
        print("#### Dummy Classifier Report")
        print(classification_report(self.y_valid, y_pred))

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(random_state=0, max_iter=3000)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Logistic Regression Report")
        print(classification_report(self.y_valid, y_pred))
        detailed_report(self.y_valid, y_pred)

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Random Forest Report")
        print(classification_report(self.y_valid, y_pred))
        detailed_report(self.y_valid, y_pred)

    def xgboost(self):
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### XGBoost Report")
        print(classification_report(self.y_valid, y_pred))
        detailed_report(self.y_valid, y_pred)

    def support_vector_machine(self):
        from sklearn.svm import SVC

        clf = SVC(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Support Vector Machine Report")
        print(classification_report(self.y_valid, y_pred))
        detailed_report(self.y_valid, y_pred)

    def k_nearest_neighbors(self):
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### K Nearest Neighbors Report")
        print(classification_report(self.y_valid, y_pred))
        detailed_report(self.y_valid, y_pred)

    def evaluate_all(self):
        self.dummy_classifier()
        self.logistic_regression()
        self.random_forest()
        self.xgboost()
        self.support_vector_machine()
        self.k_nearest_neighbors()


class TransformerBody:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer_model = AutoModel.from_pretrained(model_name).to(self.device)

    def tokenize(self, data: DatasetDict) -> DatasetDict:
        """
        :param data:
        :return:
        """
        # Tokenize the texts
        tokenized_inputs = self.tokenizer(data['visible_text'], padding="max_length", max_length=512, truncation=True,
                                          return_tensors="pt")
        return tokenized_inputs

    def extract_hidden_states(self, batch):
        """
        :param batch:
        :return:
        """
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.transformer_model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


class MarkupLMBody:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.markuplm_model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, data: DatasetDict) -> DatasetDict:
        # encoding the texts
        encoding = self.processor(data['text'], padding="max_length", max_length=512, truncation=True,
                                  return_tensors="pt")
        return encoding

    def extract_hidden_states(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.processor.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.markuplm_model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        output = torch.sigmoid(x)

        return output


class NNTrainerUtility:
    def __init__(self, device):
        self.device = device

    def train_nn_model(self, dataset: DatasetDict):
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
        criterion = torch.nn.BCELoss()
        nn_model = NeuralNetwork(input_size).to(self.device)
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
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                outputs = nn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()  # パラメータの更新
                running_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}, Training Loss: {running_loss / len(train_loader)}')
            tmp_accuracy = self.evaluate_nn_model(nn_model, valid_loader, criterion, mode="epoch")
            if tmp_accuracy > best_accuracy:
                best_accuracy = tmp_accuracy
                print(f"Best accuracy: {best_accuracy}")
                best_model = copy.deepcopy(nn_model)
            schedular.step()

        print('Finished Training')
        self.evaluate_nn_model(best_model, test_loader, criterion, mode="last")

    def evaluate_nn_model(self, nn_model: NeuralNetwork, loader: DataLoader, criterion, mode: str) -> float:
        nn_model.eval()
        all_labels = []
        all_predictions = []
        tmp_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = nn_model(inputs)
                loss = criterion(outputs, labels)
                tmp_loss += loss.item()

                # ラベルと予測結果を保存
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend((outputs > 0.5).cpu().numpy())

        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        if mode == "epoch":
            return epoch_accuracy
        elif mode == "last":
            print(f'Testing Loss: {tmp_loss / len(loader)}')
            print("#### Visible Text Classification Report ####")
            print(classification_report(all_labels, all_predictions))
            detailed_report(all_labels, all_predictions)

    def extract_outputs(self, nn_model: NeuralNetwork, dataset: DatasetDict) -> list:
        nn_model.eval()
        all_outputs = []
        test_hidden_states = dataset["testing"]["hidden_state"]
        test_dataset = TensorDataset(test_hidden_states)
        batch_size = 32
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs = data[0].to(self.device)
                outputs = nn_model(inputs)
                all_outputs.extend(outputs.cpu().numpy())
        return all_outputs
