from flask import Flask
from flask_socketio import SocketIO, emit
from threading import Thread
import time
import numpy as np
import sounddevice as sd
from collections import deque
from keras.api.models import load_model
import librosa

# Parámetros
SR = 16000
DURATION = 10
WAKE_UP = 3
MODEL_PATH = "..\models\cnn_chroma_0.1107.keras"

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

# Buffer compartido
buffer = deque(maxlen=SR * DURATION)
model = load_model(MODEL_PATH, compile=False)

def audio_callback(indata, frames, time_info, status):
    if status: print(status)
    buffer.extend(indata[:, 0])

def extract_chroma(y):
    ch = librosa.feature.chroma_stft(y=y, sr=SR)
    return np.expand_dims(ch, axis=0)

def scale_0_to_1_per_detection(X):
    # Escala cada detección individualmente al rango [0, 1]
    X_scaled = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):  # Iterar sobre cada detección (ventana)
        min_val = np.min(X[i])
        max_val = np.max(X[i])
        if max_val > min_val:  # Evitar división por cero si todos los valores son iguales
            X_scaled[i] = (X[i] - min_val) / (max_val - min_val)
        elif max_val == min_val:
            X_scaled[i] = np.zeros_like(X[i], dtype=np.float32) # O asignar un valor constante como 0.5
    return X_scaled

def preprocess_and_encode(X):
    # Normalizar (opcional, dependiendo de tus necesidades)
    normalized_X = (X - np.mean(X, axis=(1, 2), keepdims=True)) / np.std(X, axis=(1, 2), keepdims=True)

    # Escalar cada detección al rango [0, 1]
    # scaled_X = scale_0_to_1_per_detection(normalized_X)

    # Quitar la última columna y transponer
    subbed_X = normalized_X[:, :, :-1]
    Xp = subbed_X.transpose(0, 2, 1)

    # Resize
    Xp = np.resize(Xp, (Xp.shape[0], 312, 12))

    # Encoder
    encoder = load_model("..\models/encoders/encoder_chroma.keras", compile=False)
    Xc = Xp.reshape(Xp.shape + (1,))
    return encoder.predict(Xc)

def background_thread():
    """Procesa cada segundo y emite por WebSocket"""
    while True:
        time.sleep(WAKE_UP)
        if len(buffer) == SR * DURATION:
            y = np.array(buffer)
            feat = extract_chroma(y)
            XC = preprocess_and_encode(feat)
            prob = model.predict(XC)[0,0]
            label = int(prob > 0.5)
            socketio.emit("prediction", {
                "prob": float(prob),
                "label": label,
                "ts": time.strftime("%H:%M:%S")
            })

@socketio.on("connect")
def on_connect():
    emit("connected", {"msg": "Backend listo"})
    # Empieza el stream la primera vez
    global stream_thread
    if not stream_thread.is_alive():
        stream_thread.daemon = True
        stream_thread.start()

if __name__ == "__main__":
    # Inicia audio + hilo
    stream = sd.InputStream(samplerate=SR, channels=1, callback=audio_callback)
    stream.start()
    stream_thread = Thread(target=background_thread)
    socketio.run(app, host="0.0.0.0", port=5000)
