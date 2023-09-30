import json
import structlog
from structlog import get_logger
import tkinter as tk
import tkinter.scrolledtext
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from diarize_speakers import Diarization, DiarizationProcessor
from gpt import Gpt
import time
import torch
import numpy as np
import json
from transcribe import TranscribeProcessor, get_file_length
from datetime import datetime, timedelta

class Main():
    def __init__(self, config_file):
        super().__init__()
        self.logger = get_logger(__file__)
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self._gpt = Gpt(self.config)
        self.root = tk.Tk()
        # gives the window a title and dimensions
        self.root.title("Teams Postprocessing Test")
        self.root.geometry("1600x800")
        self.root.minsize(width=800, height=600)
        self.open_button = ttk.Button(self.root, text='Load a meeting recording', command=self.select_file)
        self.open_button.pack(expand=True)
        self.text = tk.Text(self.root, height=2, state='disabled')
        self.text.pack(expand=True)
        self.transcribe2_button = ttk.Button(self.root, text='Transcribe and Diarize Speakers', command=self.transcribe_fast) 
        self.transcribe2_button.pack(expand=True)
        self.speaker_text = tkinter.scrolledtext.ScrolledText(self.root, height=20, undo=True)
        self.speaker_text.pack(expand=True)  
        self.identify_button = ttk.Button(self.root, text='Identify Speakers from Context', command=self.identify) 
        self.identify_button.pack(expand=True)
        self.gpt_question = tkinter.scrolledtext.ScrolledText(self.root)
        self.gpt_question.pack(padx=5, pady=20, side=tk.LEFT)
        self.gpt_button = tk.Button(self.root, text="Ask GPT", command=self.ask_gpt)
        self.gpt_button.pack(padx=5, pady=20, side=tk.LEFT)
        self.gpt_answer = tkinter.scrolledtext.ScrolledText(self.root, state='disabled')
        self.gpt_answer.pack(padx=5, pady=20, side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.d = None
        self._transcriber = None
        self.load_last_transcription()

    def load_last_transcription(self):
        try:
            with open("transcription.txt", "r") as f:
                self.transcribed_text = f.read()
            self.update_speaker_text(self.transcribed_text)
        except Exception as e:
            self.logger.info("No previous transcription found")

    def ask_gpt(self):
        context = self.speaker_text.get(1.0, tk.END)
        question = self.gpt_question.get(1.0, tk.END)
        self.logger.info(f"Context: {context}")
        self.logger.info(f"Question: {question}")
        if len(context) == 0 or len(question) == 0:
            self.logger.info("No context or question")
            return
        answer = self._gpt.chat(context, question)
        self.logger.info(f"Answer: {answer}")
        self.gpt_answer.configure(state='normal')
        self.gpt_answer.delete(1.0, tk.END)
        self.gpt_answer.insert(tk.END, answer)
        self.gpt_answer.configure(state='disabled')

    def on_closing(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    def select_file(self):
        filetypes = (
            ('Video Recordings', '*.mp4'),
            ('All files', '*.*')
        )
        filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
        self.set_file(filename)

    def set_file(self, filename):
        self._filename = filename
        self._fileLength = get_file_length(filename)
        self.text.configure(state='normal')
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, filename)
        self.text.configure(state='disabled')
        self.logger.info(filename)
        self.logger.info(f"File length: {self._fileLength} seconds")

    def transcribe_fast(self):
        if self.d is not None or self._transcriber is not None:
            self.logger.info("Already transcribing")
            return
        self.start = time.time()
        displayed_text = "Waiting for speaker detection. Estimated finish time: "
        now = datetime.now()
        if self.config["use-gpu"] and torch.cuda.is_available():
            now += timedelta(seconds=self._fileLength/10) 
        else:
            now += timedelta(seconds=self._fileLength)
        displayed_text += f"{now.strftime('%H:%M:%S')}"
        self.update_speaker_text(displayed_text)
        self.logger.info("Starting speaker detection")
        self.d = DiarizationProcessor(self.config["huggingface-token"], self.config["use-gpu"])
        self.d._process.start()
        self.d._queue_in.put(self._filename)
        self.root.after(10, self.check_diarization_fast)

    def check_diarization_fast(self):
        try:
            self._speaker_result = self.d._queue_out.get(timeout=0)
        except Exception as e:
            self.root.after(10, self.check_diarization_fast)
            return
        self.d._queue_in.put(None)
        self.d._process.join()
        self.logger.info("starting transcription")
        self.start = time.time()
        self._transcriber = TranscribeProcessor(self.config)
        self._transcriber._process.start()
        self._transcriber.queue_in.put(self._filename)
        self._transcriber.queue_in.put((0, None))
        self._transcriber.queue_in.put(None)
        
        displayed_text = "Waiting for transcription. Estimated finish time: "
        now = datetime.now()
        if self.config["use-gpu"] and torch.cuda.is_available():
            now += timedelta(seconds=self._fileLength/10) 
        else:
            now += timedelta(seconds=self._fileLength)
        displayed_text += f"{now.strftime('%H:%M:%S')}"
        self.update_speaker_text(displayed_text)
        self.last_speaker = -1
        self.transcription_position = 0
        self.root.after(10, self.check_transcription_fast)

    def check_transcription_fast(self):
        try:
            text = self._transcriber.queue_out.get(timeout=0)
        except Exception as e:
            self.root.after(10, self.check_transcription_fast)
            return
        chunks = text["chunks"]
        self.transcribed_text = ""
        for chunk in chunks:
            start, stop = chunk["timestamp"]
            if stop is None:
                stop = len(self._speaker_result) / 100
            text = chunk["text"]
            speaker_ids = self._speaker_result[int(start*100):int(stop*100)]
            bincount = np.bincount(speaker_ids)
            speaker_id = bincount.argmax()
            if speaker_id == 0 and len(bincount) > 1:
                # get the second most common speaker
                speaker_id = bincount.argsort()[-2]
            if speaker_id != self.last_speaker:
                if len(self.transcribed_text) > 0:
                    self.transcribed_text += "\n"
                if speaker_id == 0:
                    self.transcribed_text += "Unknown Speaker: "
                else:
                    self.transcribed_text += f"Speaker {speaker_id}: "
                self.last_speaker = speaker_id
            self.transcribed_text += text
        self.update_speaker_text(self.transcribed_text)
        self.logger.debug(f"Transcription took {time.time() - self.start} seconds")
        self._transcriber._process.join()
        del self._transcriber      
        self.save_transcription()

    def save_transcription(self):
        with open("transcription.txt", "w") as f:
            f.write(self.transcribed_text)
            
    def update_speaker_text(self, text):
        self.speaker_text.delete(1.0, tk.END)
        self.speaker_text.insert(tk.END, text)

    def identify(self):
        self.transcribed_text = self.speaker_text.get(1.0, tk.END)
        if self.transcribed_text is None or len(self.transcribed_text) == 0:
            self.logger.info("No transcription result")
            return
        self.logger.info("Starting identification")
        speakers = self._gpt.speaker_detection(self.transcribed_text)
        self.logger.info(f"speakers are: {speakers}")
        if len(speakers) == 0:
            self.logger.info("No speakers identified")
            return
        querry = ""
        for speaker in speakers.keys():
            querry += f"{speaker}: {speakers[speaker]}\n"
        if tk.messagebox.askokcancel(title="Speaker Identification", message=f"Speakers identified as:\n{querry}\n\nPress OK to replace speakers"):
            for speaker in speakers.keys():
                self.transcribed_text = self.transcribed_text.replace(speaker, speakers[speaker])
            self.update_speaker_text(self.transcribed_text)


if __name__ == "__main__":
    app = Main("config.json")
    app.run()

