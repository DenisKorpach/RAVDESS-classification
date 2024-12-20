import os
import random
import shutil

# Путь к папке с отсортированными данными
sorted_path = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\sorted_by_emotion"
output_base_path = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\split_data"

# Доли данных
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Функция для разделения файлов
def split_data(emotion, files):
    random.shuffle(files)
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files

# Создание папок для Train, Validation, Test
for split in ['train', 'val', 'test']:
    for emotion in os.listdir(sorted_path):
        os.makedirs(os.path.join(output_base_path, split, emotion), exist_ok=True)

# Разделение файлов по эмоциям
for emotion in os.listdir(sorted_path):
    emotion_path = os.path.join(sorted_path, emotion)
    if os.path.isdir(emotion_path):
        files = os.listdir(emotion_path)
        train, val, test = split_data(emotion, files)

        for file, split in zip([train, val, test], ['train', 'val', 'test']):
            for f in file:
                src = os.path.join(emotion_path, f)
                dst = os.path.join(output_base_path, split, emotion, f)
                shutil.copy(src, dst)

print("Данные успешно разделены на Train, Validation и Test.")
