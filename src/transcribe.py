from collections.abc import Callable, Iterable, Mapping
import json
from typing import Any
from structlog import get_logger
import torch
import librosa
import time
import multiprocessing
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import pipeline

class TranscribeProcessor():
    def __init__(self, config):
        self.queue_in = multiprocessing.Queue()
        self.queue_out = multiprocessing.Queue()
        self.logger = get_logger(__file__)
        self.config = config
        self._process = multiprocessing.Process(target=self._run)
    
    def _run(self):
        transcriber = Transcribe(self.config)
        while True:            
            data = self.queue_in.get()
            if data is None:
                break
            try:
                if type(data) == str:
                    transcriber.load_file(data)
                else:
                    start, stop = data
                    text = transcriber.transcribe(start, stop)
                    self.queue_out.put(text)
            except Exception as e:
                self.logger.error(e)
        return


def get_file_length(filename):
    audio, sr = librosa.load(filename, sr=16000, mono = True)
    return len(audio)/sr


class Transcribe():
    def __init__(self, config):
        super().__init__()
        self.logger = get_logger(__file__)
        model = config["asr-model"]
        self.processor = AutoProcessor.from_pretrained(model, use_auth_token=config["huggingface-token"], cache_dir="models/")
        self.model = WhisperForConditionalGeneration.from_pretrained(model, use_auth_token=config["huggingface-token"] , cache_dir="models/")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model, chunk_length_s=30, stride_length_s=(5, 5),
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor, 
            generate_kwargs = {"task":"transcribe",  "language":"<|en|>"}, return_timestamps=True , device="cuda" if torch.cuda.is_available() and config["use-gpu"] else "cpu")
    
    def load_file(self, filename):
        audio, sr = librosa.load(filename, sr=16000, mono = True)
        self.audio = librosa.util.normalize(audio)
        self.sr = sr
        return
    
    def transcribe(self, start, stop):
        start_index = int(start*self.sr)
        if stop is None:
            stop_index = len(self.audio)
        else:
            stop_index = int(stop*self.sr)
        audio = self.audio[start_index:stop_index]
        prediction = self.pipe(audio, batch_size=4)
        return prediction


if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        config = json.load(f)
    EXAMPLE_MP4 = "1.mp4"
    t = Transcribe(config)
    t.load_file(EXAMPLE_MP4)
    t = t.transcribe(0, None)
    print(t)