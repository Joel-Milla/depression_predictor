# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.
from firebase_functions import firestore_fn, https_fn

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore
import json


app = initialize_app()


@https_fn.on_request()
def add_recording(req: https_fn.Request) -> https_fn.Response:
    """Handle Twilio recording status callback""" 
    # Process Twilio's data
    if req.method == "POST":
        # Get recording details from Twilio's form data
        call_sid = req.form.get("CallSid")
        recording_url = req.form.get("RecordingUrl")

        import datetime
        recording_date = datetime.datetime.now().isoformat()

        firestore_client: google.cloud.firestore.Client = firestore.client()

        doc_data = {
            "recording_url": recording_url, 
            "call_sid": call_sid, 
            "recording_date": recording_date
        }
            
        # Try to set the document
        doc_ref = firestore_client.collection("videos").document(call_sid)
        doc_ref.set(doc_data, merge=True)

        # Always return a 200 response to Twilio
        return https_fn.Response("Recording status received", status=200)

@https_fn.on_request()
def add_phone(req: https_fn.Request) -> https_fn.Response:
    """Handle Twilio recording status callback""" 
    # Process Twilio's data
    if req.method == "POST":
        # Get recording details from Twilio's form data
        call_sid = req.form.get("CallSid")
        recording_phone = req.form.get("To")

        firestore_client: google.cloud.firestore.Client = firestore.client()

        doc_data = {
            "call_sid": call_sid, 
            "recording_phone": recording_phone
        }
            
        # Try to set the document
        doc_ref = firestore_client.collection("videos").document(call_sid)
        doc_ref.set(doc_data, merge=True)

        # Always return a 200 response to Twilio
        return https_fn.Response("Recording status received", status=200)

@https_fn.on_request()
def trigger_call(req: https_fn.Request) -> https_fn.Response:
    # Get phone number from query parameters
    phone_number = req.args.get('phone_number')
    if not phone_number:
        return https_fn.Response("Missing phone number", status=400)
    
    # Remove any quotes
    phone_number = phone_number.strip('"')
    
    # Send notification to computer to make call
    db = firestore.client()
    db.collection('call_requests').add({
        'phone_number': phone_number,
        'timestamp': firestore.SERVER_TIMESTAMP,
        'status': 'pending'
    })
    
    return https_fn.Response(f"Call request for {phone_number} has been queued", status=200)