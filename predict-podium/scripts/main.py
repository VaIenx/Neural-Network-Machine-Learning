from pathlib import Path
import pandas as pd
import preprocessing
import visualizer
import model
import torch
import sys

DIR = Path(__file__).resolve().parents[1]

def predict(net, input_df, scaler, le, target):
    try:
        input_df['TeamName'] = le.transform(input_df['TeamName'])
    except ValueError:
        print(f"Unbekanntes Team. Verfügbare Teams: {list(le.classes_)}")
        return None

    # gleiche Kodierung wie in preprocessing.py
    input_df['Status']   = (input_df['Status'] == 'Finished').astype(int)
    input_df['rainfall'] = input_df['rainfall'].astype(int)

    input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.transform(
        input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']]
    )

    tensor = torch.tensor(input_df.values.astype('float32'), dtype=torch.float32)
    with torch.no_grad():
        if target == "podium":
            output = torch.sigmoid(net(tensor))
            prob = output.item()
            label = "Podium!" if prob >= 0.5 else "Kein Podium"
            print(f"  → {label} ({prob:.1%})")
        else:
            output = torch.clamp(net(tensor), 1, 20)
            position = round(output.item())
            print(f"  → Vorhergesagte Position: {position} (exakt: {output.item():.2f})")


def predict_row(net, row, scaler, le, target):
    pos = row.get('Position', '?')
    if target == 'podium':
        if isinstance(pos, (int, float)):
            true_display = f"P{int(pos)} → {'Podium' if pos <= 3 else 'Kein Podium'}"
        else:
            true_display = '?'
    else:
        true_display = pos

    print(f"  Fahrer: {row.get('Abbreviation','?')}  |  "
          f"Team: {row.get('TeamName','?')}  |  "
          f"Grid: {row.get('GridPosition','?')}  |  "
          f"Rennen: {row.get('race','?')} {row.get('year','')}  |  "
          f"Echter Wert: {true_display}")

    input_df = pd.DataFrame([{
        'TeamName':       row['TeamName'],
        'GridPosition':   row['GridPosition'],
        'Status':         row['Status'],
        'box':            row['box'],
        'median_laptime': row['median_laptime'],
        'Q_best_sec':     row['Q_best_sec'],
        'rainfall':       row['rainfall'],
    }])
    predict(net, input_df, scaler, le, target)


def custom_input(net, scaler, le, target):
    while True:
        print("\n--- Custom Input ---")
        try:
            grid   = int(input("  Startplatz [1-20] >> "))
            team   = input("  Team [Ferrari/McLaren/...] >> ")
            status = int(input("  Finished [0/1] >> "))
            box    = int(input("  Box [0-4] >> "))
            median = float(input("  Median Laptime [z.B. 100] >> "))
            bestq  = float(input("  Best Q time [z.B. 100] >> "))
            rain   = int(input("  Rain [0/1] >> "))
        except ValueError:
            print("Ungültige Eingabe, bitte nochmal.")
            continue

        # Status als int → 'Finished' simulieren damit predict() konsistent bleibt
        input_df = pd.DataFrame([{
            'TeamName':       team,
            'GridPosition':   grid,
            'Status':         'Finished' if status == 1 else 'Retired',
            'box':            box,
            'median_laptime': median,
            'Q_best_sec':     bestq,
            'rainfall':       rain,
        }])
        predict(net, input_df, scaler, le, target)

        if input("\nNochmal? [j/n] >> ").strip().lower() != 'j':
            break


def training_data_input(net, df_raw, scaler, le, target):
    true_col = 'Podium' if target == 'Podium' else 'Position'

    while True:
        print("\n--- Trainingsdaten testen ---")
        sample = df_raw.sample(5).reset_index(drop=True)

        print(f"  {'#':<3} {'Fahrer':<8} {'Team':<20} {'Grid':<6} {'Rennen':<30} {'Echter Wert'}")
        print("  " + "-" * 80)
        for i, row in sample.iterrows():
            rennen = f"{row.get('race','?')} {row.get('year','')}"
            print(f"  [{i+1}] {str(row.get('Abbreviation','?')):<8} "
                  f"{str(row.get('TeamName','?')):<20} "
                  f"{str(row.get('GridPosition','?')):<6} "
                  f"{rennen:<30} "
                  f"{row.get(true_col,'?')}")

        print("\n  [1-5] Zeile vorhersagen   [n] Neue Auswahl   [0] Zurück")
        choice = input("  >> ").strip().lower()

        if choice == '0':
            break
        elif choice == 'n':
            continue
        elif choice in ('1', '2', '3', '4', '5'):
            row = sample.iloc[int(choice) - 1]
            print()
            predict_row(net, row, scaler, le, target)
            input("\n  [Enter] Weiter >> ")
        else:
            print("  Ungültige Eingabe.")


if __name__ == "__main__":
    target = input("Modus [podium/position] >> ").strip().lower()
    if target not in ("podium", "position"):
        print("Ungültiger Modus.")
        sys.exit(1)

    df_raw = pd.read_csv(f'{DIR}/DATA.csv')
    viz    = visualizer.Visualizer(df_raw, save_dir='predict-podium/plots')

    X_train, X_test, y_train, y_test, scaler, le = preprocessing.preprocessing_data(target=target)
    epochs = 150 if target == 'podium' else 800
    net    = model.PodiumNet(X_train, X_test, y_train, y_test, epochs=epochs, target=target)
    net.run()
    loss_list, val_loss_list = net.evaluate()
    viz.plot_training(loss_list, val_loss_list)

    while True:
        print("\nEingabemodus:")
        print("  [1] Custom Input")
        print("  [2] Trainingsdaten testen")
        print("  [0] Beenden")
        choice = input(">> ").strip()

        if choice == '1':
            custom_input(net, scaler, le, target)
        elif choice == '2':
            training_data_input(net, df_raw, scaler, le, target)
        elif choice == '0':
            break
        else:
            print("Ungültige Eingabe.")