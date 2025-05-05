from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load model dan scaler
model = tf.keras.models.load_model('model_lstm.h5')
scaler = joblib.load('scaler.save')

time_step = 5  # harus sama dengan saat training

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting list of OHLC dicts [{'open':..., 'high':..., 'low':..., 'close':...}, ...]
        df = pd.DataFrame(data)
        if df.shape[0] < time_step:
            return jsonify({'error': f'Input data harus minimal {time_step} baris OHLC'}), 400

        # Normalisasi data
        scaled = scaler.transform(df[['open', 'high', 'low', 'close']])
        X = scaled[-time_step:]  # ambil time_step terakhir
        X = X.reshape(1, time_step, 4)

        # Prediksi
        pred_scaled = model.predict(X)
        # Inverse transform hanya untuk kolom close (index 3)
        dummy = np.zeros((pred_scaled.shape[0], 4))
        dummy[:, 3] = pred_scaled[:, 0]
        pred_price = scaler.inverse_transform(dummy)[0, 3]

        return jsonify({'predicted_close': float(pred_price)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
