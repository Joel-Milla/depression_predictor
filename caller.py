import os
import json
import base64
import asyncio
import argparse
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
import websockets
from dotenv import load_dotenv
import uvicorn
import re

load_dotenv()

# Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
PHONE_NUMBER_FROM = os.getenv('PHONE_NUMBER_FROM')
RECORDING_CALLBACK_URL= os.getenv('RECORDING_CALLBACK_URL')
PHONE_CALLBACK_URL= os.getenv('PHONE_CALLBACK_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
raw_domain = os.getenv('DOMAIN', '')
DOMAIN = re.sub(r'(^\w+:|^)\/\/|\/+$', '', raw_domain) # Strip protocols and trailing slashes from DOMAIN

PORT = int(os.getenv('PORT', 6060))
SYSTEM_MESSAGE = (
    "You are a calm, empathetic AI assistant whose primary purpose is to check in on the person's wellbeing. "
    "Your tone is warm, gentle, and understanding. Ask thoughtful questions about how they're feeling, "
    "both emotionally and physically. Listen attentively to their responses and acknowledge their feelings "
    "without judgment. Don't rush to offer solutions - focus on being present and understanding their perspective. "
    "Express genuine interest in their day-to-day experiences and challenges. If they seem distressed, "
    "respond with compassion and validation. Keep conversations focused on their wellbeing rather than "
    "providing facts or information unless specifically requested."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI(debug=True)

if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and PHONE_NUMBER_FROM and OPENAI_API_KEY):
    raise ValueError('Missing Twilio and/or OpenAI environment variables. Please set them in the .env file.')

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.get('/', response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.websocket('/media-stream')
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        extra_headers=[
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "realtime=v1")
        ]
    ) as openai_ws:
        await initialize_session(openai_ws)
        stream_sid = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)
                    if response['type'] == 'session.updated':
                        print("Session updated successfully:", response)
                    if response['type'] == 'response.audio.delta' and response.get('delta'):
                        try:
                            audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            await websocket.send_json(audio_delta)
                        except Exception as e:
                            print(f"Error processing audio data: {e}")
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation so AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Do not talk, just wait and listen to me"
                    )
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Have the AI speak first
    await send_initial_conversation_item(openai_ws)

async def check_number_allowed(to):
    """Check if a number is allowed to be called."""
    try:
        incoming_numbers = client.incoming_phone_numbers.list(phone_number=to)
        if incoming_numbers:
            return True

        outgoing_caller_ids = client.outgoing_caller_ids.list(phone_number=to)
        if outgoing_caller_ids:
            return True

        return False
    except Exception as e:
        print(f"Error checking phone number: {e}")
        return False

async def make_call(phone_number_to_call: str):
    """Make an outbound call."""
    if not phone_number_to_call:
        raise ValueError("Please provide a phone number to call.")

    # is_allowed = await check_number_allowed(phone_number_to_call)
    is_allowed = True
    
    if not is_allowed:
        raise ValueError(f"The number {phone_number_to_call} is not recognized as a valid outgoing number or caller ID.")

    outbound_twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{DOMAIN}/media-stream" /></Connect></Response>'
    )

    call = client.calls.create(
        record=True,
        status_callback=PHONE_CALLBACK_URL,
        recording_status_callback=RECORDING_CALLBACK_URL,
        from_=PHONE_NUMBER_FROM,
        to=phone_number_to_call,
        twiml=outbound_twiml
    )

    await log_call_sid(call.sid)

async def log_call_sid(call_sid):
    """Log the call SID."""
    print(f"Call started with SID: {call_sid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Twilio AI voice assistant server.")
    parser.add_argument('--call', required=True, help="The phone number to call, e.g., '--call=+18005551212'")
    args = parser.parse_args()

    phone_number = args.call

    loop = asyncio.get_event_loop()
    loop.run_until_complete(make_call(phone_number))
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
