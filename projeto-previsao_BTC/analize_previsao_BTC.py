from flask import Flask, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import io
import base64
import threading
import time

app = Flask(__name__)

# Variáveis globais para armazenar o modelo e os dados
modelo, scaler, df, X_test, y_test = None, None, None, None, None

# Função para carregar os dados do Yahoo Finance e preparar o modelo
def carregar_modelo():
    global modelo, scaler, df, X_test, y_test

    print("[INFO] Atualizando modelo...")
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    df = yf.download('BTC-USD', start='2021-01-01', end=end_date, progress=False)
    df = df[['Close']]

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[['Close']].values)

    # Criar dataset para treino
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(data_scaled, time_step)

    # Dividir treino e teste
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Ajustar formato para LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Criar modelo LSTM
    modelo = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)  # Menos epochs para rodar mais rápido

    print("[INFO] Modelo atualizado com sucesso!")

# Função para atualizar o modelo a cada 5 minutos
def atualizar_modelo_periodicamente():
    while True:
        carregar_modelo()
        time.sleep(300)  # Espera 5 minutos (300 segundos)

# Iniciar a atualização automática em uma thread separada
threading.Thread(target=atualizar_modelo_periodicamente, daemon=True).start()

# Rota para previsão do próximo dia
@app.route("/previsao")
def previsao():
    if modelo is None:
        return jsonify({"erro": "Modelo ainda não carregado"}), 500

    last_sequence = X_test[-1:].reshape(1, X_test.shape[1], 1)
    predicted_value = modelo.predict(last_sequence)[0][0]

    predicted_price = scaler.inverse_transform([[predicted_value]])[0][0]
    last_real_price = float(df['Close'].iloc[-1])
    recomendacao = "COMPRA" if predicted_price > last_real_price else "VENDA"

    return jsonify({
        "preco_anterior": round(last_real_price, 2),
        "preco_previsto": round(predicted_price, 2),
        "recomendacao": recomendacao
    })

# Rota para exibir gráfico como imagem
@app.route("/grafico")
def grafico():
    if modelo is None:
        return "Modelo ainda não carregado", 500

    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Preço Real', color='blue')
    
    predicted_price = previsao().json["preco_previsto"]
    plt.axhline(y=predicted_price, color='red', linestyle='--', label='Previsão Próximo Dia')
    
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Previsão de Preço do Bitcoin')
    plt.legend()
    
    # Converter gráfico para base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f'<img src="data:image/png;base64,{plot_url}">'

# Página principal
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)