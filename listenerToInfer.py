import firebase_admin
from firebase_admin import credentials, firestore
import subprocess
import time
import requests
from twilio.rest import Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize Firebase
cred = credentials.Certificate('../ai-caller-9c525-firebase-adminsdk-fbsvc-a5e5317ad8.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def on_snapshot(_, changes, __):
    for change in changes:
        # Only process new or modified documents
        if change.type.name in ['ADDED', 'MODIFIED']:
            doc = change.document
            data = doc.to_dict()
            
            # Look for Twilio recording URL fields
            # These typically start with https://api.twilio.com/
            twilio_url = None
            
            for key, value in data.items():
                if isinstance(value, str) and value.startswith("https://api.twilio.com/"):
                    twilio_url = value
                    print(f"Found Twilio URL: {twilio_url}")
                    
                    # Process the recording
                    process_recording(doc.id, twilio_url)
                    break
            
            if not twilio_url:
                print("No Twilio URL found in this document")

def process_recording(doc_id, twilio_url):
    """Process a Twilio recording URL"""
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        response = requests.get(
            f"{twilio_url}.mp3", 
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        
        if response.status_code == 200:
            with open(f"{doc_id}.mp3", "wb") as f:
                f.write(response.content)
            
            print(f"Recording saved as {doc_id}.mp3")
            
            # call model
            # analyze_audio(f"{doc_id}.mp3")
            
            # Update the document to indicate processing is complete
            db.collection('videos').document(doc_id).update({
                'processed': True,
                'local_file': f"{doc_id}.mp3"
            })
        else:
            print(f"Failed to download recording: {response.status_code}")
            
    except Exception as e:
        print(f"Error processing recording: {e}")

# Listen for changes on videos
videos_ref = db.collection('videos')
query_watch = videos_ref.on_snapshot(on_snapshot)

print("Listening for new recordings in the 'videos' collection...")

# Keep program running
while True:
    time.sleep(1)