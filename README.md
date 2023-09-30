# TeamsMeetingPostProcessing
Video Transcription, speaker detection and GPT chat support program, intended for speeding up the postprocessing prozess of recorded workshops.

## Installation
1. Install Git und Python (https://git-scm.com/download/mac https://www.python.org/downloads/macos/)
2. git clone https://github.com/Elbrasch/TeamsMeetingPostProcessing
3. On Windows: Download ffmpeg and place ffmpeg.exe, ffplay.exe and ffprobe.exe in src (or make the executables available in windows PATH Variable)
4. rename src/config_template.json to src/config.json, add values (whisper transcription model on huggingface and microsoft cloud cognitive services GPT information)
5. Open Shell/Terminal
6. Optional: Install pytorch with GPU acceleration for 10x faster transcription, dependent on your platform
7. pip install -r requirements.txt
8. cd src
9. python main.py

## How to get a huggingface token
1. Create a hf.co user
2. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
3. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)