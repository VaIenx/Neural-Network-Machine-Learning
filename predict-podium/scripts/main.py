from pathlib import Path
import pandas as pd
import preprocessing
import visualizer
import model
import torch
import os
import sys

DIR = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    # Modus abfragen und validieren
    target = input("Modus [podium/position] >> ").strip().lower()
    if target not in ("podium", "position"):
        print("Ungültiger Modus. Bitte 'podium' oder 'position' eingeben.")
        sys.exit(1)

    df = pd.read_csv(f'{DIR}/DATA.csv')
    viz = visualizer.Visualizer(df, save_dir='predict-podium/plots')

    # Daten vorverarbeiten, Modell erstellen, trainieren und auswerten
    X_train, X_test, y_train, y_test, scaler, le = preprocessing.preprocessing_data(target=target)
    if target == 'podium':
        epochs = 150
    elif target == 'position':
        epochs = 800
    net = model.PodiumNet(X_train, X_test, y_train, y_test, epochs=epochs, target=target)
    net.run()
    loss_list, val_loss_list = net.evaluate()
    viz.plot_training(loss_list, val_loss_list)

    def UserInput():
        # Fahrerdaten vom Nutzer einlesen
        grid = int(input("Startplatz [1-20] >> "))
        team = input("Team [Ferrari/McLaren/...] >> ")
        status = int(input("Finished: [0/1] >> "))
        box = int(input("Box [0-4] >> "))
        medianlaptime = int(input("Median Laptime [100] >>"))
        bestQsec = int(input("best Q time [100] >>"))
        rain = int(input("Rain [0/1] >> "))

        # Eingabe als DataFrame aufbereiten
        input_df = pd.DataFrame([{
            'TeamName': team,
            'GridPosition': grid,
            'Status': status,
            'box': box,
            'median_laptime': medianlaptime,
            'Q_best_sec': bestQsec,
            'rainfall': rain
        }])

        # Gleiche Kodierung wie beim Training anwenden
        input_df['TeamName'] = le.transform(input_df['TeamName'])
        input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.transform(
            input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']]
        )

        # Vorhersage
        tensor = torch.tensor(input_df.values, dtype=torch.float32)
        with torch.no_grad():
            if target == "podium":
                # Sigmoid → Wahrscheinlichkeit, Schwellwert 0.5
                output = torch.sigmoid(net(tensor))
                prob = output.item()
                if prob >= 0.5:
                    print(f"Podium! ({prob:.1%})")
                else:
                    print(f"Kein Podium ({prob:.1%})")
            else:
                # Ausgabe auf gültige Positionen 1–20 begrenzen
                output = torch.clamp(net(tensor), 1, 20)
                position = round(output.item())
                print(f"Vorhergesagte Position: {position} (exakt: {output.item():.2f})")
        input("Enter >>")

    # Endlosschleife für wiederholte Eingaben
    while True:
        # os.system('cls')
        UserInput()