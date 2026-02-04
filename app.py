
import streamlit as st
import joblib

st.title("Прогноз стоимости недвижимости")
features = joblib.load('model_features.pkl')
model = joblib.load('house_price_model.pkl')

input_data = {}

for feature in features:
    if feature == 'total_square':
        val = st.number_input("Площадь (м²)", min_value=10, max_value=1100, value=10, step=1)
    elif feature == 'rooms':
        val = st.number_input("Количество комнат", min_value=1, max_value=6, value=2, step=1)
    elif feature == 'floor':
        val = st.number_input("Этаж", min_value=1, max_value=53, value=1, step=1)
    else:
        val = st.number_input(f"Значение для '{feature}'", value=0.0, step=0.1)
    input_data[feature] = val

if st.button("Рассчитать стоимость"):
    X_input = [[input_data[f] for f in features]]
    prediction = model.predict(X_input)[0]


    st.success(f"Прогнозируемая стоимость: **{prediction:,.0f} руб.**")
