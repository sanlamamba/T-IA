# Speech to Text Using Vosk

This repository provides a simple speech-to-text application using the Vosk model. The application allows you to convert speech into text either in real time using a microphone or from a pre-recorded `.wav` audio file.

## Features

- **STT Streaming (Microphone-based)**: Listen and transcribe speech in real time using your microphone.
- **STT File-based**: Transcribe speech from an audio file (`.wav` format) with a sample rate of 16,000 Hz and mono channel.

## Context

### What is Vosk?
[Vosk](https://alphacephei.com/vosk/) is an open-source offline speech recognition toolkit that allows for real-time transcription. It supports multiple languages and platforms, including Linux, Windows, macOS, and mobile devices.

### Downloading the Model
Download the Vosk model for the language you need from the [Vosk models page](https://alphacephei.com/vosk/models). Choose a model according to the language you want to transcribe.

### Model Directory
After downloading, place the model in a `models` folder within the project directory. Update the `model_path` in the code accordingly.

## Requirements

- Python 3.6 or higher
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio streaming
- [Vosk](https://alphacephei.com/vosk/)

## Installation

1. Clone the repository:
   ```
      git clone https://github.com/sanlamamba/T-AIA-901_par_1.git
   ```

2. Download the Vosk model for the desired language and place it in the `models` folder.

3. Create a virtual environment and install the dependencies from the `requirements.txt` file:
   ```
      python -m venv venv
      source venv/bin/activate   # On Windows use `venv\Scripts\activate`
      pip install -r requirements.txt
   ```

4. Install PyAudio (if not already installed):
   ```
      pip install pyaudio
   ```

## Usage

### STT Streaming (Microphone-based)

To convert speech to text in real time using your microphone, run the following command:

     ```
          python stt_streaming.py
     ```

The script listens through the microphone, converting the spoken words into text in real time.

**Note**: Ensure that your microphone is properly configured and connected before running the script.

### STT File-based

To convert speech from an audio file to text, run the following command:

     ```
          python stt_file.py
     ```

This script reads an audio file and outputs the transcription. By default, the audio file should:
- Be in `.wav` format.
- Have a sample rate of 16,000 Hz.
- Be a mono channel.

You can convert your audio file to the appropriate format using an online converter, such as [Online WAV Converter](https://audio.online-convert.com/convert-to-wav).

**Example**:
- Place your audio file in the project directory (e.g., `audio.wav`).
- The output will be saved to `output.txt`.

## Code Overview

### STT Streaming (`stt_streaming.py`)

This script uses a Vosk model to recognize speech from a live audio stream:

- Loads the Vosk model.
- Uses PyAudio to capture microphone input.
- Transcribes the captured audio into text and prints it in real time.

Key parts of the code:
- `model_path`: Specify the path to your Vosk model.
- `KaldiRecognizer`: Uses the Vosk model to recognize the speech.
- Outputs partial and full transcriptions.

### STT File-based (`stt_file.py`)

This script processes a pre-recorded `.wav` audio file to convert speech into text:

- Loads the Vosk model.
- Reads the `.wav` audio file using the `wave` library.
- Recognizes speech using `KaldiRecognizer`.
- Outputs the recognized text to the console and saves it to an output file.

Key parts of the code:
- `model_path`: Path to your Vosk model.
- `audio_file_path`: Path to the `.wav` file that you want to transcribe.
- `output_file_path`: Path to save the transcription.

## Notes

- The default Vosk models may have limited accuracy for certain languages and specific accents. You can try different models or train your own to improve accuracy.
- The audio file must be mono with a sample rate of 16,000 Hz for optimal performance.
- You may adjust the file paths and model paths within the script according to your local environment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to [Vosk](https://alphacephei.com/vosk/) for the amazing open-source speech recognition toolkit.
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for making audio processing with Python easier.
