# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.
from firebase_functions import firestore_fn, https_fn

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

app = initialize_app()


@https_fn.on_request()
def add(req: https_fn.Request) -> https_fn.Response:
    """Handle Twilio recording status callback"""
    
    # Print all data from the request for debugging
    print("Request method:", req.method)
    print("Request args:", req.args)
    print("Request form:", req.form)
    print("Request json:", req.get_json(silent=True))
    
    # Process Twilio's data (they typically send as form data)
    if req.method == "POST":
        # Get recording details from Twilio's form data
        recording_sid = req.form.get("RecordingSid")
        recording_url = req.form.get("RecordingUrl")
        recording_status = req.form.get("RecordingStatus")
        
        print(f"Recording SID: {recording_sid}")
        print(f"Recording URL: {recording_url}")
        print(f"Recording Status: {recording_status}")

        firestore_client: google.cloud.firestore.Client = firestore.client()

        # Push the new message into Cloud Firestore using the Firebase Admin SDK.
        _, doc_ref = firestore_client.collection("videos").add({recording_sid: recording_url
        })
        
        # Here you could store the recording info in Firestore if needed
        
        # Always return a 200 response to Twilio
        return https_fn.Response("Recording status received", status=200)