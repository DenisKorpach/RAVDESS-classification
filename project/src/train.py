import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === Paths and Configuration ===
BASE_PATH = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\split_data"
OUTPUT_PATH = r"D:\STUDING_DENZA\4KYRS\MRZIS_KURSUCK\code\dataset\prepared_data"
PKL_PATH = os.path.join(OUTPUT_PATH, 'ravdess_data.pkl')
EMOTIONS_MAP = {
    "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
    "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
}
MFCC_FEATURES = 40
MAX_PAD_LEN = 216

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Feature Extraction ===
def extract_features(file_path, n_mfcc=MFCC_FEATURES, max_pad_len=MAX_PAD_LEN):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max(0, max_pad_len - mfccs.shape[1])
    return np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')[:, :max_pad_len]

# === Data Preparation ===
def prepare_data(split):
    data, labels = [], []
    split_path = os.path.join(BASE_PATH, split)
    for emotion, label in EMOTIONS_MAP.items():
        emotion_path = os.path.join(split_path, emotion)
        if os.path.isdir(emotion_path):
            for file in os.listdir(emotion_path):
                file_path = os.path.join(emotion_path, file)
                data.append(extract_features(file_path))
                labels.append(label)
    return np.array(data), np.array(labels)

# === Data Loading or Processing ===
def load_or_prepare_data():
    if not os.path.exists(PKL_PATH):
        print("Preparing data...")
        data_dict = {}
        for split in ['train', 'val', 'test']:
            print(f"Preparing {split} data...")
            data_dict[split] = prepare_data(split)
        with open(PKL_PATH, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        print("Loading prepared data...")
        with open(PKL_PATH, 'rb') as f:
            data_dict = pickle.load(f)
    return data_dict

# === Normalize and Reshape Data ===
def preprocess_data(data_dict):
    for key in data_dict:
        data, labels = data_dict[key]
        data = data.reshape((data.shape[0], data.shape[2], data.shape[1]))
        data /= np.max(data)
        data_dict[key] = (data, labels)
    return data_dict

# === Model Definition ===
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, 5, padding='same', input_shape=input_shape),
        Activation('relu'),
        Conv1D(128, 5, padding='same'),
        Activation('relu'),
        Dropout(0.1),
        MaxPooling1D(pool_size=8),
        Conv1D(128, 5, padding='same'),
        Activation('relu'),
        Conv1D(128, 5, padding='same'),
        Activation('relu'),
        Dropout(0.2),
        Flatten(),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Real-Time Plot Callback ===
class RealTimePlot(Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.train_accuracy.append(logs['accuracy'])
        self.val_accuracy.append(logs['val_accuracy'])

        # Update plots
        self.ax[0].cla()
        self.ax[0].plot(self.epochs, self.train_loss, label='Train Loss')
        self.ax[0].plot(self.epochs, self.val_loss, label='Validation Loss')
        self.ax[0].set_title('Loss During Training')
        self.ax[0].set_xlabel('Epochs')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()

        self.ax[1].cla()
        self.ax[1].plot(self.epochs, self.train_accuracy, label='Train Accuracy')
        self.ax[1].plot(self.epochs, self.val_accuracy, label='Validation Accuracy')
        self.ax[1].set_title('Accuracy During Training')
        self.ax[1].set_xlabel('Epochs')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].legend()

        plt.pause(0.1)

    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    data_dict = load_or_prepare_data()
    data_dict = preprocess_data(data_dict)

    X_train, y_train = data_dict['train']
    X_val, y_val = data_dict['val']
    X_test, y_test = data_dict['test']

    input_shape = X_train.shape[1:]
    num_classes = len(EMOTIONS_MAP)
    model = build_model(input_shape, num_classes)

    real_time_plot = RealTimePlot()
    model.fit(X_train, y_train, epochs=700, batch_size=16, validation_data=(X_val, y_val), callbacks=[real_time_plot])

    # Save the trained model
    model_save_path = os.path.join(OUTPUT_PATH, 'trained_emotion_model.h5')
    model.save(model_save_path)
    print(f"Trained model saved at: {model_save_path}")

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=list(EMOTIONS_MAP.keys())))

    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(EMOTIONS_MAP.keys()), yticklabels=list(EMOTIONS_MAP.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
