import torch
import librosa
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from google.cloud import firestore
import sys
import json

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

emotions = ['anger', 'disgust', 'fear', 'happy', 'pleasant_surprised', 'sad', 'neutral']

def load_model(model_path, num_classes):
    print("Loading model...")
    model = EmotionRecognitionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully")
    return model

def predict_emotion(model, audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    interval_duration = 2  # seconds
    interval_samples = interval_duration * sr
    num_intervals = len(y) // interval_samples

    results = []

    for i in range(num_intervals):
        start_sample = i * interval_samples
        end_sample = start_sample + interval_samples
        y_interval = y[start_sample:end_sample]

        mel_spectrogram = librosa.feature.melspectrogram(y=y_interval, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        max_length = 128
        pad_width = max_length - mel_spectrogram_db.shape[1]
        if pad_width > 0:
            mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spectrogram_db = mel_spectrogram_db[:, :max_length]
        mel_spectrogram_3ch = np.repeat(mel_spectrogram_db[np.newaxis, :, :], 3, axis=0)
        input_tensor = torch.tensor(mel_spectrogram_3ch, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy().flatten()
            predicted_class = output.argmax(dim=1).item()

        probability_dict = {emotions[j]: f"{prob:.4f}" for j, prob in enumerate(probabilities)}

        result = {
            "interval": i + 1,
            "predicted_emotion": emotions[predicted_class],
            "probability_distribution": probability_dict
        }
        results.append(result)

    return results

if __name__ == "__main__":
    model_path = '/home/shengbin/dlweek/depression_predictor/ml/models/emotion_recognition_model.pth'
    audio_file_path = sys.argv[1]
    
    model = load_model(model_path, num_classes=len(emotions))
    results = predict_emotion(model, audio_file_path)
    print(json.dumps(results, indent=4))