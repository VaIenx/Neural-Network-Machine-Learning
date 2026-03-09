import fastf1, os
import pandas as pd
from tqdm import tqdm
import visualizer


class FastF1:
    def __init__(self):
        if not os.path.exists('cache'): os.makedirs('cache')
        fastf1.Cache.enable_cache('cache')  # Verhindert wiederholte API-Calls bei erneutem Laden

        self.races_to_load = [
            (2021, 'Belgium'),
            (2021, 'France'),
            (2021, 'Austria'),
            (2021, 'Silverstone'),
            (2021, 'Abu Dhabi')
        ]

        self.teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren']
        self.all_laps_data = []

    def load_newDataSet(self):
        final_df = pd.DataFrame()

        for year, grandprix in self.races_to_load:
            print(f"\n--- Lade {grandprix} {year} ---")
            try:
                session = fastf1.get_session(year, grandprix, 'R')
                session.load()

                # Boxenrunden raus – die verfälschen die LapTime stark
                laps = session.laps.pick_teams(self.teams).pick_wo_box()

                if len(laps) == 0:
                    print(f"Keine Daten für {grandprix} gefunden. Überspringe...")
                    continue

                max_speeds, min_speeds, valid_indices = [], [], []

                for index, lap in tqdm(laps.iterrows(), total=len(laps), desc=f"Telemetrie {grandprix}"):
                    # Nur Runden unter grüner Flagge – kein Safety Car, kein VSC
                    if str(lap['TrackStatus']) == '1':
                        try:
                            car_data = lap.get_car_data()

                            if len(car_data) > 0:
                                max_speeds.append(car_data['Speed'].max())
                                min_speeds.append(car_data['Speed'].min())
                                valid_indices.append(index)
                            else:
                                continue
                        except Exception as e:
                            print(f"Übersprungen: {e}")
                            continue

                race_data = laps.loc[valid_indices].copy()
                race_data['MaxSpeed'] = max_speeds
                race_data['MinSpeed'] = min_speeds
                race_data['GP'] = grandprix

                self.all_laps_data.append(race_data)

            except Exception as e:
                print(f"Fehler beim Laden von {grandprix}: {e}")

        if self.all_laps_data:
            final_df = pd.concat(self.all_laps_data, ignore_index=True)

            # Nur relevante Spalten behalten – alles andere ist für das NN nicht nötig
            keep_cols = ['LapTime', 'LapNumber', 'TyreLife', 'Compound', 'Team', 'MaxSpeed', 'MinSpeed', 'GP']
            final_df = final_df[keep_cols]

            # LapTime von timedelta (0:01:31.456) in Sekunden (91.456) umwandeln
            final_df['LapTime'] = pd.to_timedelta(final_df['LapTime']).dt.total_seconds()

            # Runden mit fehlenden Werten entfernen – NNs können kein NaN verarbeiten
            final_df = final_df.dropna()

            final_df.to_csv('DATA.csv', index=False)
            print(f"\nERFOLG: {len(final_df)} Runden aus {len(self.races_to_load)} Rennen in DATA.csv gespeichert!")
        else:
            print("Keine Daten extrahiert.")

        return final_df


class NeuronalNetwork:
    def __init__(self):
        pass

    def set_df(self, DataFrame):
        self.__F1_df = DataFrame

    def get_df(self):
        return self.__F1_df

    def print_df(self):
        print(self.__F1_df)
    
    


class MAIN:
    def __init__(self):
        F1 = FastF1()
        self.NeuronalNetwork = NeuronalNetwork()

        if os.path.exists('DATA.csv'):
            if str(input('Es existiert bereits eine Datenbank. Neu laden und überschreiben? (Y/N) ')).upper() == 'Y':
                self.NeuronalNetwork.set_df(F1.load_newDataSet())
            else:
                # Bestehende CSV laden statt neu von der API zu holen (spart 10-15 Minuten)
                self.NeuronalNetwork.set_df(pd.read_csv('DATA.csv'))
                print("Runden aus DATA.csv geladen.")
        else:
            self.NeuronalNetwork.set_df(F1.load_newDataSet())

        self.NeuronalNetwork.print_df()
        viz = visualizer.Visualizer(self.NeuronalNetwork.get_df(), "./plots")
        viz.plot_all() 


if __name__ == '__main__':
    MAIN()