import os
import sys
import wave
import json
from vosk import Model, KaldiRecognizer

# Path to your model directory and audio file
model_path = "./models/vosk-model-small-fr-0.22"
audio_file_path = "./audio.wav"
output_file_path = "./output.txt"

# Load the Vosk model
if not os.path.exists(model_path):
    print(f"Model path '{model_path}' does not exist")
    sys.exit(1)

if not os.path.exists(audio_file_path):
    print(f"Audio file '{audio_file_path}' does not exist")
    sys.exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

# Open the audio file
wf = wave.open(audio_file_path, "rb")
if wf.getnchannels() != 1:
    print("Only mono audio files are supported. Please provide a mono WAV file.")
    sys.exit(1)

# Process the audio file
recognized_text = ""

print("Processing the audio file...")

while True:
    data = wf.readframes(4096)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        text = json.loads(result)["text"]
        if text:
            recognized_text += text + "\n"
            print(f"Recognized: {text}")
    else:
        partial_result = recognizer.PartialResult()
        partial_text = json.loads(partial_result)["partial"]
        if partial_text:
            print(f"Partial: {partial_text}")

with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(recognized_text)

print(f"\nProcessing complete. Output saved to '{output_file_path}'.")

# Close the audio file
wf.close()
