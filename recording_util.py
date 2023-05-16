import datetime
import os
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import logging

class RecordingUtil():

    def __init__(self):
        self._counter = 0
        current_time = datetime.datetime.now()
        self._recording_path = f"{os.path.expanduser('~')}/Downloads/Voice_Team/{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}"
        logging.info(f"Create folder: {self._recording_path}")
        Path(self._recording_path).mkdir(parents=True, exist_ok=True)


    def record_wav(self, audio, sample_rate):
        self._counter    = self._counter + 1
        if type(audio) != np.ndarray:
            audio          = audio.numpy()
        audio          = np.int16(audio / np.max(np.abs(audio)) * 32767)  # scale
        full_fill_name = f"{self._recording_path}/{self._counter}.wav"
        print(f"Record to file: {full_fill_name}")
        write(full_fill_name, sample_rate, audio)

