import os
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import librosa
import torch
from torchvision.models import VGG16_Weights

class EmotionDataset(Dataset):
    def __init__(self, data_path, emotions, transform=None):
        self.data_path = data_path
        self.emotions = emotions
        self.file_list = []
        self.labels = []
        self.transform = transform

        for idx, emotion in enumerate(emotions):
            emotion_folders = [f'YAF_{emotion}', f'OAF_{emotion}']
            for folder in emotion_folders:
                folder_path = os.path.join(data_path, folder)
                if os.path.exists(folder_path):
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        self.file_list.append(file_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file_path, sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        max_length = 128
        pad_width = max_length - mel_spectrogram_db.shape[1]
        if pad_width > 0:
            mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spectrogram_db = mel_spectrogram_db[:, :max_length]
        mel_spectrogram_3ch = np.repeat(mel_spectrogram_db[np.newaxis, :, :], 3, axis=0)
        return torch.tensor(mel_spectrogram_3ch, dtype=torch.float32), torch.tensor(label)

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
data_path = './data'
dataset = EmotionDataset(data_path, emotions)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


model = EmotionRecognitionModel(num_classes=len(emotions))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


total_train_correct = 0
total_train_samples = 0
total_val_correct = 0
total_val_samples = 0


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = total_train_correct / total_train_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            total_val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = total_val_correct / total_val_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


final_train_accuracy = total_train_correct / total_train_samples
final_val_accuracy = total_val_correct / total_val_samples
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")


torch.save(model.state_dict(), 'emotion_recognition_model.pth')

test_loader = DataLoader(test_dataset, batch_size=32)

model.load_state_dict(torch.load('emotion_recognition_model.pth'))

model.eval()
test_loss = 0.0
total_test_correct = 0
total_test_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        total_test_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_test_samples += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = total_test_correct / total_test_samples
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def predict_emotion(audio_path):
    
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  
    max_length = 128
    pad_width = max_length - mel_spectrogram_db.shape[1]
    if pad_width > 0:
        mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_length]
    mel_spectrogram_3ch = np.repeat(mel_spectrogram_db[np.newaxis, :, :], 3, axis=0)
    input_tensor = torch.tensor(mel_spectrogram_3ch, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return emotions[predicted_class]

audio_file_path = '/home/shengbin/dlweek/custom/data/OAF_Fear/OAF_back_fear.wav'  # Replace with your audio file path
predicted_emotion = predict_emotion(audio_file_path)
print(f'Predicted Emotion: {predicted_emotion}')
