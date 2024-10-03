import os
import sys
import wave
import json
import pyaudio
from vosk import Model, KaldiRecognizer

# Path to your model directory
model_path = "./models/vosk-model-small-fr-0.22"

# Load the Vosk model
if not os.path.exists(model_path):
    print(f"Model path '{model_path}' does not exist")
    sys.exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

print("Listening...")

try:
    while True:
        try:
            data = stream.read(4096, exception_on_overflow=False)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = json.loads(result)["text"]
                if text:
                    print(f"Recognized: {text}")
            else:
                partial_result = recognizer.PartialResult()
                partial_text = json.loads(partial_result)["partial"]
                if partial_text:
                    print(f"Partial: {partial_text}")
        except OSError as e:
            print(f"Warning: {e}")
except KeyboardInterrupt:
    print("Terminating...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
