import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('predict-podium/DATA.csv')

df['podium'] = (df['Position'] <= 3).astype(int) # Wenn podium dann 1
df = df.drop(columns=['Abbreviation', 'Position', 'year', 'race']) # spalten rauswerfen damit das NN nicht cheated
df['Status'] = (df['Status'] == 'Finished').astype(int) # Wenn finished dann 1
df['rainfall'] = df['rainfall'].astype(int) # regen statt Bool als 0/1

le = LabelEncoder() # codiert die Namen in zahlen
df['TeamName'] = le.fit_transform(df['TeamName'])


scaler = StandardScaler() # skaliert die Zahlen, damit alle gleich vieli wert sind
df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']] = scaler.fit_transform(df[['GridPosition', 'box', 'median_laptime', 'Q_best_sec']]) 


# SPLITTING
X = df.drop(columns=['podium'])
y = df['podium']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

print(df.head())
print(df.dtypes)
