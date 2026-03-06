import fastf1, os
import pandas as pd
from tqdm import tqdm

if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2021, 'France', 'R')
session.load()

# 1. Nur die Spitzenteams auswählen
top_teams = ['Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren'] 
laps = session.laps.pick_teams(top_teams).pick_wo_box()

max_speeds = []
min_speeds = []

print(f"Extrahiere Telemetrie für {len(laps)} Runden...")

# 2. Telemetrie-Loop mit Fortschrittsbalken
# tqdm(laps.iterrows(), total=len(laps)) erzeugt den Balken
for index, lap in tqdm(laps.iterrows(), total=len(laps), desc="Fortschritt"):
    try:
        telemetry = lap.get_telemetry()
        max_speeds.append(telemetry['Speed'].max())
        min_speeds.append(telemetry['Speed'].min())
    except:
        max_speeds.append(None)
        min_speeds.append(None)

# 3. Daten zusammenführen
laps['MaxSpeed'] = max_speeds
laps['MinSpeed'] = min_speeds

f1_data = pd.DataFrame(laps)
keep_cols = ['LapTime', 'TyreLife', 'Compound', 'Team', 'MaxSpeed', 'MinSpeed']
f1_data = f1_data[keep_cols]

# 4. Datenbereinigung
f1_data['LapTime'] = pd.to_timedelta(f1_data['LapTime']).dt.total_seconds()
f1_data = f1_data.dropna()

# 5. Export
f1_data.to_csv('DATA.csv', index=False)
print("\nExport abgeschlossen: DATA.csv")