
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv('realty_data.csv')

features = ['total_square']+["rooms"]+["floor"]
target = 'price'
df = df[features + [target]].dropna()
x = df[features]
y = df[target]

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
model = RandomForestRegressor(n_estimators=100, random_state=10)
model.fit(x_train, y_train)

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(features, 'model_features.pkl')

print("Модель обучена и сохранена!")
print(f"Используемые признаки: {features}")

