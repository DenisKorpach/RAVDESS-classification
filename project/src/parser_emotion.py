import os
import shutil

# Путь к датасету
dataset_path = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\audio_speech_actors_01-24"

# Путь для сохранения по эмоциям
output_path = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\sorted_by_emotion"

# Словарь для сопоставления эмоций
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Создание папок для каждой эмоции
for emotion in emotions.values():
    emotion_path = os.path.join(output_path, emotion)
    os.makedirs(emotion_path, exist_ok=True)

# Перебор всех файлов в папках актеров
for actor_folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_path):
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                # Извлечение идентификатора эмоции
                emotion_id = filename.split("-")[2]
                emotion = emotions.get(emotion_id)
                if emotion:
                    # Копирование файла в соответствующую папку
                    source_path = os.path.join(actor_path, filename)
                    dest_path = os.path.join(output_path, emotion, filename)
                    shutil.copy(source_path, dest_path)

print("Файлы успешно распределены по папкам эмоций.")
