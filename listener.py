# Example client using Firestore listener
import firebase_admin
from firebase_admin import credentials, firestore
import subprocess
import time

# Initialize Firebase
cred = credentials.Certificate('../ai-caller-9c525-firebase-adminsdk-fbsvc-a5e5317ad8.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        data = doc.to_dict()
        if data['status'] == 'pending':
            phone_number = data['phone_number']
            print(f"Received request to call {phone_number}")
            
            # Execute your call script
            subprocess.run(["python", "caller.py", "--call", phone_number])
            
            # Update status
            doc.reference.update({'status': 'processing'})

# Listen for new call requests
call_requests_ref = db.collection('call_requests')
query = call_requests_ref.where('status', '==', 'pending')
query_watch = query.on_snapshot(on_snapshot)

# Keep program running
while True:
    time.sleep(1)