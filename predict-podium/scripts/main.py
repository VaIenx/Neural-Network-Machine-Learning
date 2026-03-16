import pandas as pd
import preprocessing
import visualizer

if __name__ == "__main__":
    df = pd.read_csv('predict-podium/DATA.csv')
    viz = visualizer.Visualizer(df, save_dir='predict-podium/plots')
    viz.plot_all()

    preprocessing.run()