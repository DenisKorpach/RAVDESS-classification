import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from pydub.playback import play
import threading

# === Константы ===
MODEL_PATH = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\prepared_data\trained_emotion_model.h5"
EMOTIONS_MAP = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}
MFCC_FEATURES = 40
MAX_PAD_LEN = 216

# === Загрузка модели ===
try:
    model = load_model(MODEL_PATH)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()

# === Функции ===
def extract_features(file_path, n_mfcc=MFCC_FEATURES, max_pad_len=MAX_PAD_LEN):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max(0, max_pad_len - mfccs.shape[1])
        return np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_len]
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка обработки файла: {e}")
        return None

def classify_audio(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, features.shape[1], features.shape[0])
        features = features / np.max(features)  # Нормализация
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return EMOTIONS_MAP[predicted_class]
    return None

def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка воспроизведения: {e}")

def on_open_file():
    file_path = filedialog.askopenfilename(
        title="Выберите аудиофайл",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    if not file_path:
        return

    # Воспроизведение аудио в отдельном потоке
    threading.Thread(target=play_audio, args=(file_path,), daemon=True).start()

    # Классификация и вывод результата
    true_emotion = os.path.basename(os.path.dirname(file_path))  # Предполагаем, что истинная эмоция — имя папки
    predicted_emotion = classify_audio(file_path)

    if predicted_emotion:
        result_label.config(text=f"Результат классификации: {predicted_emotion}\nИстинное значение: {true_emotion}")
    else:
        result_label.config(text="Ошибка классификации.")

# === Интерфейс пользователя ===
root = tk.Tk()
root.title("Классификация эмоций по аудиофайлу")
root.geometry("500x300")

# Заголовок
title_label = tk.Label(root, text="Классификация эмоций по аудио", font=("Arial", 16))
title_label.pack(pady=10)

# Кнопка выбора файла
open_file_button = tk.Button(root, text="Выбрать аудиофайл", command=on_open_file, font=("Arial", 12))
open_file_button.pack(pady=20)

# Результат классификации
result_label = tk.Label(root, text="Результат появится здесь", font=("Arial", 12))
result_label.pack(pady=10)

# Запуск приложения
root.mainloop()
