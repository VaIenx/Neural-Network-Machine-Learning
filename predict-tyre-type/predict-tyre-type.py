import fastf1, os
import pandas as pd

if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2021, 'France', 'R')
session.load()

laps = session.laps

f1_data = pd.DataFrame(laps)

print(f1_data.head())
print(f"\nDatensatz-Größe: {f1_data.shape}")