from pydub.utils                       import mediainfo
from huggingface_hub.hf_api            import HfFolder

import soundfile                       as sf
import librosa
import logging

def get_sample_rate(file):
    info      = mediainfo(file)
    return int(info['sample_rate'])

def get_wav_duration(wav_file):
    f = sf.SoundFile(wav_file)
    return f.frames / f.samplerate


def save_last_30_sec(source_wav, dest_wav):

    speech, sr = librosa.load(source_wav, sr=16000)
    if 30 * 16000 > len(speech):
        sf.write(dest_wav, speech, sr)
    else:
        start_index = len(speech) - 30*16000
        speech = speech[start_index:]
        sf.write(dest_wav, speech, sr)

def convert_to_16sr_file(source_path, dest_path):
    speech, sr = librosa.load(source_path, sr=16000)
    sf.write(dest_path, speech, sr)
    return speech

def save_huggingface_token():
    logging.info("Save hugging face token")
    MY_TOKEN = "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"
    HfFolder.save_token(MY_TOKEN)