import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder 
import joblib

data = pd.read_csv('data/medii_bucuresti.csv', delimiter=';')

le = LabelEncoder()
data['Profil_Code'] = le.fit_transform(data['Profil'])

X = data[['Profil_Code', 'Ultima_Medie']]

y = data['Ultima_Medie']

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model/model.pkl')

joblib.dump(le, 'model/label_encoder.pkl')

print("Modelul a fost salvat cu succes!")
