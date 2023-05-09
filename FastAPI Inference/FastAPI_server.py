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


# uvicorn.run(app, host='0.0.0.0', port=8888)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)