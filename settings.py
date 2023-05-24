import logging

import torch
import os
import whisper
#from pyannote.audio import Pipeline

#
#   General
#
DEVICE             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE        = 16000
SAVE_RESULTS_PATH  = f"{os.getcwd()}/TmpFiles"
record_to_wav      = False
MAX_SAVED_RESULTS  = 5



#
#   whisper server
#
RAMBO_IP           = "10.53.140.33:86"
SERVER_IP          = RAMBO_IP
WHISPER_URL        = f'http://{SERVER_IP}/gradio_demo_live/'
#WHISPER_URL        = f'http://10.53.140.230:8123/relay/'
LANGUAGES          = whisper.tokenizer.LANGUAGES
FILTER_BAD_RESULUS = False

#
#   whisper thresholds and settings
#
compression_ratio_threshold = 2.4
logprob_threshold           = -1.0
no_speech_threshold         = 0.95
user_prompt                 = False
use_language                = None

#
#   local whisper
#
RUN_LOCAL_WHISPER  = True
whisper_model      = None
whisper_model_type = "base" # for local running

#
#   offline diarization
#
sd_pipeline          = None #Pipeline.from_pretrained("pyannote/speaker-diarization")
run_online_pyyannote = True

