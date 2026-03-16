import fastf1
import pandas as pd
from tqdm import tqdm
import logging

fastf1.Cache.enable_cache('./cache')
logging.getLogger('fastf1').setLevel(logging.ERROR)

class FastF1Collector:
    def __init__(self, years: list):
        self._years = years
        self._races = {}
        for year in self._years:
            self._races[year] = self.get_races_for_year(year)

        self._df = pd.DataFrame()
        self.create_DataFrame()

    def get_races_for_year(self, year):
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        races = schedule['EventName'].tolist()
        print(f"{year}: {len(races)} Rennen gefunden")
        return races

    def get_session_data(self, year, race):
        session = fastf1.get_session(year, race, "R")
        session.load(weather=True)
        weather = session.weather_data

        data = session.results[['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status']].copy()

        data['Status'] = data['Status'].apply(
            lambda x: 'DNF' if any(
                word in str(x) for word in ['Crash', 'Engine', 'Gearbox', 'Hydraulics', 'Brakes', 'Suspension', 'Retired', 'Accident', 'Collision', 'Power', 'Electrical']
            ) else 'Finished'
        )

        stints_per_driver = session.laps.groupby('Driver')['Stint'].max() - 1
        data['box'] = data['Abbreviation'].map(stints_per_driver)

        median_laptimes = (
            session.laps
            .pick_wo_box()
            .groupby('Driver')['LapTime']
            .median()
            .dt.total_seconds()
        )
        data['median_laptime'] = data['Abbreviation'].map(median_laptimes)

        quali = fastf1.get_session(year, race, 'Q')
        quali.load()
        def best_quali_time(row):
            for col in ['Q3', 'Q2', 'Q1']:
                val = row[col]
                if pd.notna(val):
                    return pd.to_timedelta(val).total_seconds()
            return None
        quali_times = quali.results[['Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
        quali_times['Q_best_sec'] = quali_times.apply(best_quali_time, axis=1)
        data = data.merge(quali_times[['Abbreviation', 'Q_best_sec']], on='Abbreviation', how='left')

        data['year'] = year
        data['race'] = race
        data['rainfall'] = bool(weather['Rainfall'].any())

        return data

    def append_data_to_DataFrame(self, data):
        self._df = pd.concat([self._df, data], ignore_index=True)

    def create_DataFrame(self):
        for year, races in self._races.items():
            with tqdm(total=len(races), desc=f"{year}", unit="GP", position=0, leave=True, dynamic_ncols=True) as pbar:
                for race in races:
                    pbar.set_postfix({"Rennen": race[:12]})
                    try:
                        data = self.get_session_data(year, race)
                        self.append_data_to_DataFrame(data)
                    except Exception as e:
                        tqdm.write(f"✗ {race} {year}: {e}")
                    pbar.update(1)
        

        self._df = self._df.sort_values('year').reset_index(drop=True)

    def save_to_csv(self, path='DATA.csv'):
        if self._df.empty:
            print("DataFrame ist leer – nichts gespeichert.")
            return
        self._df = self._df.dropna()
        self._df.to_csv(path, index=False)
        print(f"Gespeichert: {path} ({len(self._df)} Zeilen)")

if __name__ == "__main__":
    collector = FastF1Collector(years=[2022, 2023, 2024])
    collector.save_to_csv()