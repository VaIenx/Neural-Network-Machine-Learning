import warnings
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class PodiumNet(nn.Module):
    def __init__(self, X_train, X_test, y_train, y_test, epochs, target="podium"):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.epochs = epochs
        self.target = target  # "podium" = Klassifikation, "position" = Regression
        self.loss_list = []
        self.val_loss_list = []

        # Netzarchitektur: 7 Eingabe-Features → 64 → 32 → 16 → 1 Ausgabe
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

        if self.target == "podium":
            self.fc4 = nn.Linear(16, 1)  # Binärklassifikation
        else:
            self.fc4 = nn.Linear(16, 1)  # Regression (gleiche Architektur, anderer Loss)

    def forward(self, x):
        # Vorwärtsdurchlauf mit ReLU-Aktivierung nach jedem Hidden Layer
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

    def evaluate(self):
        self.eval()  # Dropout/BatchNorm in Inferenzmodus schalten
        X_test_t = torch.tensor(self.X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            if self.target == "podium":
                # Sigmoid → Wahrscheinlichkeit → binäre Vorhersage
                outputs = torch.sigmoid(self(X_test_t))
                predicted = (outputs >= 0.5).float()
                accuracy = (predicted == y_test_t).float().mean()
                print(f"Accuracy: {accuracy.item():.4f}")
                self.accuracy = accuracy.item()
            else:
                # Ausgabe auf 1–20 clippen, MAE als Metrik
                outputs = torch.clamp(self(X_test_t), 1, 20)
                mae = (outputs - y_test_t).abs().float().mean()
                print(f"MAE: {mae.item():.4f} Positionen")
                self.mae = mae.item()
        return self.loss_list, self.val_loss_list

    def run(self, verbose=1):
        if self.target == "podium":
            # Klassengewicht ausgleichen, da Podiumsplätze seltener sind (3 von 20)
            pos = (self.y_train == 1).sum()
            neg = (self.y_train == 0).sum()
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # MSE für Regression auf kontinuierliche Positionswerte
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        X_train_t = torch.tensor(self.X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1)

        bar = tqdm(range(self.epochs), desc="Training", unit="epoch", disable=verbose == 0)

        for epoch in bar:
            self.train()  # Trainingsmodus aktivieren
            optimizer.zero_grad()
            outputs = self(X_train_t)       # Vorwärtsdurchlauf
            loss = criterion(outputs, y_train_t)  # Loss berechnen
            loss.backward()                 # Gradienten berechnen
            optimizer.step()               # Gewichte anpassen

            # Validierungsloss ohne Gradientenberechnung ermitteln
            with torch.no_grad():
                X_test_t = torch.tensor(self.X_test.values, dtype=torch.float32)
                y_test_t = torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1)
                val_outputs = self(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                self.val_loss_list.append(val_loss.item())
            self.loss_list.append(loss.item())

            # Live-Anzeige in der Fortschrittsbar
            bar.set_postfix(loss=f"{loss.item():.4f}", val_loss=f"{val_loss.item():.4f}")

            if verbose >= 2 and epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")