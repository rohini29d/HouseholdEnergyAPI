import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    df = pd.read_csv("energy_data.csv", parse_dates=[['Date', 'Time']])
    df.set_index('Date_Time', inplace=True)
    df['hour'] = df.index.hour

    q_low = df['Global_active_power'].quantile(0.01)
    q_hi = df['Global_active_power'].quantile(0.99)
    df_filtered = df[(df['Global_active_power'] > q_low) & (df['Global_active_power'] < q_hi)]

    features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour']
    target = 'Global_active_power'

    X = df_filtered[features]
    y = df_filtered[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    with open("energy_model.pkl", "wb") as f:
        pickle.dump(model, f)

    result_df = pd.DataFrame({"Actual": y, "Predicted": preds})

    return mse, r2, result_df

def predict(input_data):
    with open("energy_model.pkl", "rb") as f:
        model = pickle.load(f)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return prediction
