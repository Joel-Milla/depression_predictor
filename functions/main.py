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
        recording_phone = req.form.get("CallSid")

        import datetime
        recording_date = datetime.datetime.now().isoformat()

        firestore_client: google.cloud.firestore.Client = firestore.client()

        # Push the new message into Cloud Firestore using the Firebase Admin SDK.
        _, doc_ref = firestore_client.collection("videos").document(recording_sid).set({"recording_url": recording_url, "recording_sid": recording_sid, "recording_date": recording_date, "recording_phone": recording_phone
        })
                
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