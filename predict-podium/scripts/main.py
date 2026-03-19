from pathlib import Path
import pandas as pd
import preprocessing
import visualizer
import model
import torch
import os

DIR = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df = pd.read_csv(f'{DIR}/DATA.csv')
    viz = visualizer.Visualizer(df, save_dir='predict-podium/plots')
    #viz.plot_all()

    X_train, X_test, y_train, y_test, scaler, le = preprocessing.preprocessing_data()
    net = model.PodiumNet(X_train, X_test, y_train, y_test, epochs=130)
    net.run()
    loss_list, val_loss_list = net.evaluate()
    viz.plot_training(loss_list, val_loss_list)


    def UserInput():
        grid = int(input("Startplatz [1-20] >> "))
        team = input("Team [Ferrari/McLaren/...] >> ")
        status = int(input("Finished: [0/1] >> "))
        box = int(input("Box [0-4] >> "))
        medianlaptime = int(input("Median Laptime [100] >>"))
        bestQsec = int(input("best Q time [100] >>"))
        rain = int(input("Rain [0/1] >> "))

        input_df = pd.DataFrame([{
            'TeamName': team,
            'GridPosition': grid,
            'Status': status,
            'box': box,
            'median_laptime': medianlaptime,
            'Q_best_sec': bestQsec,
            'rainfall': rain
        }])
        input_df['TeamName'] = le.transform(input_df['TeamName'])

        input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.transform(input_df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']])

        tensor = torch.tensor(input_df.values, dtype=torch.float32)
        with torch.no_grad():
            output = torch.sigmoid(net(tensor))
            prob = output.item()
            if prob >= 0.5:
                print(f"Podium! ({prob:.1%})")
            else:
                print(f"Kein Podium ({prob:.1%})")
    while True:
        os.system('cls')
        UserInput()