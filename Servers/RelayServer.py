import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import json
import numpy as np
import requests

os.environ["TIKTOKEN_CACHE_DIR"] = "cache"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"])  # Allows all headers




@app.get("/relay")
def get_transcription(value):

    print("GOT RELAY REQUEST")
    RAMBO_IP = "10.53.140.33:86"
    SERVER_IP = RAMBO_IP
    WHISPER_URL = f'http://{SERVER_IP}/gradio_demo_live/'

    print("Send to rambo")
    res = requests.get(WHISPER_URL, json=value)
    print("Got from rambo")
    return res


# uvicorn.run(app, host='0.0.0.0', port=8888)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8123)
    #uvicorn.run(app, host="0.0.0.0", port=8123)