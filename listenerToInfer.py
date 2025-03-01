import firebase_admin
from firebase_admin import credentials, firestore
import subprocess
import time
import requests
from twilio.rest import Client
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
import librosa
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
import json

# Load environment variables
load_dotenv()

# Get Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize Firebase
cred = credentials.Certificate('../ai-caller-9c525-firebase-adminsdk-fbsvc-a5e5317ad8.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

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
    interval_duration = 5  # seconds
    interval_samples = interval_duration * sr
    num_intervals = len(y) // interval_samples

    intervalResults = []
    overall_probabilities = np.zeros(len(emotions))

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

        overall_probabilities += probabilities

        probability_dict = {emotions[j]: f"{prob:.4f}" for j, prob in enumerate(probabilities)}

        intervalResult = {
            "interval": i + 1,
            "predicted_emotion": emotions[predicted_class],
            "probability_distribution": probability_dict
        }
        intervalResults.append(intervalResult)

    # Normalize overall probabilities
    overall_probabilities /= num_intervals
    overall_distribution = {emotions[j]: f"{prob:.4f}" for j, prob in enumerate(overall_probabilities)}

    print(json.dumps(intervalResults, indent=2))
    print(json.dumps(overall_distribution, indent=2))

    return intervalResults, overall_distribution

def on_snapshot(_, changes, __):
    for change in changes:
        # Only process new or modified documents
        if change.type.name in ['ADDED', 'MODIFIED']:
            doc = change.document
            data = doc.to_dict()
            
            # Look for Twilio recording URL fields
            # These typically start with https://api.twilio.com/
            twilio_url = None
            recording_date = None
            
            for key, value in data.items():
                if key == "recording_date":
                    recording_date = value
                    print(f"Recording date: {value}")
                    
                if isinstance(value, str) and value.startswith("https://api.twilio.com/"):
                    twilio_url = value
                    print(f"Found Twilio URL: {twilio_url}")
                    
                    # Process the recording
                    process_recording(doc.id, twilio_url, recording_date)
                    break
            
            if not twilio_url:
                print("No Twilio URL found in this document")

def process_recording(doc_id, twilio_url, recording_date):
    """Process a Twilio recording URL"""
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        response = requests.get(
            f"{twilio_url}.mp3", 
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        
        if response.status_code == 200:
            mp3_path = f"./mp3_files/{doc_id}.mp3"
            wav_path = f"./wav_files/{doc_id}.wav"
            
            with open(mp3_path, "wb") as f:
                f.write(response.content)
            
            print(f"Recording saved as {mp3_path}")
            
            # Convert mp3 to wav
            # audio = AudioSegment.from_mp3(mp3_path)
            # audio.export(wav_path, format="wav")
            # print(f"Converted {mp3_path} to {wav_path}")
            # Convert mp3 to wav using ffmpeg
            subprocess.run(['ffmpeg', '-i', mp3_path, wav_path], check=True)
            print(f"Converted {mp3_path} to {wav_path}")
            
            # Perform model inference
            perform_inference(doc_id, wav_path, recording_date)
            
            # Optionally, delete the mp3 file after conversion
            os.remove(mp3_path)
        else:
            print(f"Failed to download recording: {response.status_code}")
            
    except Exception as e:
        print(f"Error processing recording: {e}")

def perform_inference(doc_id, audio_path, recording_date):
    """Perform model inference on an audio file"""
    try:
        model_path = '/home/shengbin/dlweek/depression_predictor/ml/models/emotion_recognition_model.pth'
        model = load_model(model_path, num_classes=len(emotions))
        intervalResults, overallResults = predict_emotion(model, audio_path)
        
        # Update Firestore with the inference results and recording date
        db.collection('results').document(doc_id).set({
            'recording_date': recording_date,
            'overallResults': overallResults,
            'intervalResults': intervalResults
        })
        print(f"Inference results saved to Firestore for document {doc_id}")
        
    except Exception as e:
        print(f"Error performing inference: {e}")

# Listen for changes on videos
videos_ref = db.collection('videos')
query_watch = videos_ref.on_snapshot(on_snapshot)

print("Listening for new recordings in the 'videos' collection...")

# Keep program running
while True:
    time.sleep(1)