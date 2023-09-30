from pyannote.audio import Pipeline
from pydub import AudioSegment, effects  
import time
import math
import numpy as np
from structlog import get_logger
import os
import torch
import multiprocessing

class DiarizationProcessor():
    def __init__(self, use_auth_token:str, use_gpu) -> None:
        self._queue_in = multiprocessing.Queue()
        self._queue_out = multiprocessing.Queue()
        self._process = multiprocessing.Process(target=self._run)
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu if torch.cuda.is_available() else False

    def _run(self):
        diarizer = Diarization(self.use_auth_token, self.use_gpu)
        while True:
            data = self._queue_in.get()
            if data is None:
                break
            d = diarizer.diarize_speakers(data)
            self._queue_out.put(d[0])


class Diarization:
    def __init__(self, use_auth_token: str, use_gpu) -> None:
        self.logger = get_logger(__file__)
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1.1", use_auth_token=use_auth_token)
        # Try to use GPU if available
        self.use_gpu = use_gpu
        if self.use_gpu:
            try:
                self.pipeline.to(torch.device("cuda"))
            except Exception as e:
                self.logger.warning(
                    f"Could not use GPU for speaker diarization: {e}")
                pass

    def diarize_speakers(self, path_to_mp4_file: str) -> np.ndarray:
        # diarizes the speakers with https://huggingface.co/pyannote/speaker-diarization
        if self.use_gpu:
            self.pipeline.to(torch.device("cuda"))
        start = time.time()
        sound = self.load_wav_from_mp4(path_to_mp4_file)
        self.logger.info(f"Loaded audio in {time.time() - start:.1f} seconds.")
        start = time.time()
        tmp = path_to_mp4_file.split("/")[-1]
        sound_file = "diarize_" + ".".join(tmp.split(".")[:-1])+ ".wav"
        sound.export(sound_file, format="wav")
        self.logger.info(
            f"Exported audio in {time.time() - start:.1f} seconds.")
        start = time.time()
        diarization = self.pipeline(sound_file)
        self.logger.info(
            f"Extracted Speakers from audio in {time.time() - start:.1f} seconds.")
        if self.use_gpu:
            self.pipeline.to(torch.device("cpu"))
        self.logger.info(diarization)
        os.remove(sound_file)
        diarized_speakers = np.array(
            [0 for i in range(math.ceil(sound.duration_seconds)*100)])
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            for i in range(math.floor(turn.start*100), math.ceil(turn.end*100)):
                speaker_id = int(speaker.split("_")[-1])
                diarized_speakers[i] = speaker_id + 1
        return diarized_speakers, diarization

    @staticmethod
    def load_wav_from_mp4(path_to_mp4_file: str) -> AudioSegment:
        # loads the audio from an mp4 video file and sets 16kHz and mono
        sound = AudioSegment.from_file(path_to_mp4_file)
        sound = sound.set_frame_rate(int(16 * 1e3))
        sound = sound.set_channels(1)
        sound = effects.normalize(sound)
        return sound


if __name__ == "__main__":
    # 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
    # 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
    # 3. instantiate pretrained speaker diarization pipeline
    import json
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
    with open("config.json", "r") as f:
        config = json.load(f)
    USE_AUTH_TOKEN = config["huggingface-token"]
    EXAMPLE_MP4 = "../data/Kickoff - John Hopkins University-20230817_170450-Meeting Recording.mp4"
    model = Diarization(USE_AUTH_TOKEN)
    diarized_speakers, diarization = model.diarize_speakers(EXAMPLE_MP4)

