from flask import Flask, jsonify, request
from keras.models import load_model
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

# Muat model
model = load_model('model.h5')

# Buat API untuk memprediksi harga coin
@app.route('/predict', methods=['GET'])
def predict():
    # Ambil data dari GekkoAPI
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=30"
    headers = {
        "x-cg-demo-api-key": "CG-7pi9DCcf6E6PmCFBLrwvGtZT"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Preprocessing data
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close']] = scaler.fit_transform(df[['open', 'high', 'low', 'close']])
    
    # Buat dataset untuk prediksi
    X = df[['open', 'high', 'low', 'close']].values[-7:]
    X = X.reshape((1, 7, 4))
    
    # Prediksi harga coin
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(np.array([[0, 0, 0, prediction[0][0]]]))[:, 3][0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
