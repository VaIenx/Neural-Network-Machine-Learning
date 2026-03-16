import warnings
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class PodiumNet(nn.Module):
    def __init__(self, X_train, X_test, y_train, y_test, epochs):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.epochs = epochs
        self.loss_list = []
        self.val_loss_list = []

        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x
    
    def evaluate(self):
        self.eval()
        X_test_t = torch.tensor(self.X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():       # kein Gradient nötig bei Evaluation
            outputs = torch.sigmoid(self(X_test_t))
            predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test_t).float().mean()
        print(f"Accuracy: {accuracy.item():.4f}")
        self.accuracy = accuracy.item()
        return self.loss_list, self.val_loss_list

    def run(self, verbose=1):  # verbose: 0=nichts, 1=progressbar, 2=progressbar+epoch logs
        pos = (self.y_train == 1).sum()
        neg = (self.y_train == 0).sum()
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        X_train_t = torch.tensor(self.X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1)

        bar = tqdm(range(self.epochs), desc="Training", unit="epoch", disable=verbose == 0)

        for epoch in bar:
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train_t)  # Vorhersage
            loss = criterion(outputs, y_train_t)  # Loss berechnen
            loss.backward()  # Backpropagation
            optimizer.step()  # Gewichte anpassen
            with torch.no_grad():
                X_test_t = torch.tensor(self.X_test.values, dtype=torch.float32)
                y_test_t = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)
                val_outputs = self(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                self.val_loss_list.append(val_loss.item())
            self.loss_list.append(loss.item())

            bar.set_postfix(loss=f"{loss.item():.4f}", val_loss=f"{val_loss.item():.4f}")  # live update in bar

            if verbose >= 2 and epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                