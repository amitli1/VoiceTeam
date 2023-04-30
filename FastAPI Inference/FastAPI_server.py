from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import whisper
import numpy as np


def fastapi_main(host, port, size='medium', device='cuda:0'):
    audio_model = whisper.load_model(size, device)
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"],  # Allows all origins
                       allow_credentials=True,
                       allow_methods=["*"],  # Allows all methods
                       allow_headers=["*"])  # Allows all headers

    class WhisperInput(BaseModel):
        wav: list
        languages: list

    class WhisperInputLive(BaseModel):
        wav: list

    @app.get("/gradio_demo_static")
    def get_transcription(whisper_: WhisperInput):
        wav = [np.float(i) for i in whisper_.wav]
        audio = whisper.pad_or_trim(np.array(wav)).astype('float32')
        print(type(audio), type(audio[0]))
        mel = whisper.log_mel_spectrogram(audio).to('cuda')
        options = whisper.DecodingOptions(fp16=False, task='transcribe')
        result = whisper.decode(audio_model, mel, options)
        no_speech_prob = result.no_speech_prob
        mel = whisper.log_mel_spectrogram(audio).to('cuda')

        _, probs = audio_model.detect_language(mel)

        temp_lang = max(probs, key=probs.get)
        print(result)
        print(result.text, "no_speech_prob: ", no_speech_prob)
        trnscrb = ''
        if no_speech_prob < 0.6:
            trnscrb = result.text + ' '
            whisper_.languages.append(temp_lang)
            print(temp_lang)
            if len(whisper_.languages) > 3:
                whisper_.languages.pop(0)
        return [trnscrb, whisper_.languages]

    @app.get("/gradio_demo_live")
    def get_transcription(whisper_: WhisperInputLive):
        wav = [np.float(i) for i in whisper_.wav]
        audio = whisper.pad_or_trim(np.array(wav)).astype('float32')
        mel = whisper.log_mel_spectrogram(audio).to('cuda')
        options = whisper.DecodingOptions(fp16=False, task='transcribe', beam_size=5)
        result = whisper.decode(audio_model, mel, options)
        return [result]

    uvicorn.run(app, host=host, port=port)
