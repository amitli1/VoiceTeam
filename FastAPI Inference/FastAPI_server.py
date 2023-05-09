from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import whisper
import torch
import json
import numpy as np


audio_model = whisper.load_model('large', 'cuda')
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],  # Allows all origins
                    allow_credentials=True,
                    allow_methods=["*"],  # Allows all methods
                    allow_headers=["*"])  # Allows all headers

class WhisperInput(BaseModel):
    wav: list
    languages: list
    prompt: list

class WhisperInputLive(BaseModel):
    wav: list

class WhisperBatchInput(BaseModel):
    wav: str
    languages: list

@app.get("/gradio_demo_static")
def get_transcription(whisper_: WhisperInput):
    wav = [np.float(i) for i in whisper_.wav]
    audio = whisper.pad_or_trim(np.array(wav)).astype('float32')
    # print(type(audio), type(audio[0]))
    mel = whisper.log_mel_spectrogram(audio).to('cuda')
    options = whisper.DecodingOptions(fp16=False, task='transcribe')
    result = whisper.decode(audio_model, mel, options)
    no_speech_prob = result.no_speech_prob
    mel = whisper.log_mel_spectrogram(audio).to('cuda')

    _, probs = audio_model.detect_language(mel)

    temp_lang = max(probs, key=probs.get)
    # print(result)
    # print(result.text, "no_speech_prob: ", no_speech_prob)
    trnscrb = ''
    if no_speech_prob < 0.6:
        trnscrb = result.text + ' '
        whisper_.languages.append(temp_lang)
        # print(temp_lang)
        if len(whisper_.languages) > 3:
            whisper_.languages.pop(0)
    return [trnscrb, whisper_.languages]

@app.get("/gradio_demo_live")
def get_transcription(whisper_: WhisperInput):
    # print('LIVE')/
    wav = [float(i) for i in whisper_.wav]
    audio = whisper.pad_or_trim(np.array(wav)).astype('float32')
    mel = whisper.log_mel_spectrogram(audio).to('cuda')
    prompt = whisper_.prompt[0]
    if len(prompt) == 0:
        prompt = None
    
    options = whisper.DecodingOptions(fp16=True, task='transcribe', beam_size=5, language=whisper_.languages[0], prompt = prompt)
    result = whisper.decode(audio_model, mel, options)
    # print(result)
    return [result]

@app.get("/batch_inference")
def get_transcription(whisper_: WhisperBatchInput):
    # print('LIVE')/
    # wav = [np.float(i) for i in whisper_.wav]
    wav = np.array(json.loads(whisper_.wav))
    wav = torch.from_numpy(wav.astype('float32')).to("cuda")
    options = whisper.DecodingOptions(task='transcribe', beam_size=None)
    result = whisper.decode(audio_model, wav, options)
    return [result]

@app.get("/batch_inference_beam")
def get_transcription(whisper_: WhisperBatchInput):
    # print('LIVE')/
    # wav = [np.float(i) for i in whisper_.wav]
    wav = np.array(json.loads(whisper_.wav))
    wav = torch.from_numpy(wav.astype('float32')).to("cuda")
    options = whisper.DecodingOptions(task='transcribe', beam_size=5)
    result = whisper.decode(audio_model, wav, options)
    return [result]

uvicorn.run(app, host='0.0.0.0', port=8888)