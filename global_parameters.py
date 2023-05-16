import torch
import queue
from recording_util   import RecordingUtil
import pandas         as pd

#
#   VAD
#
vad_model, vad_utils   = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(vad_get_speech_timestamps, vad_save_audio, vad_read_audio, VADIterator, vad_collect_chunks) = vad_utils

vad_speech_model, vad_utils2               = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(speech_get_speech_timestamps, _, _, _, _) = vad_utils2
vad_iterator                               = VADIterator(vad_model)

last_30_sec_vad      = None
total_num_of_vad_elm = 0
last_vad_plot        = None

#
# Queues
#
speech_queue    = queue.Queue()
processed_queue = queue.Queue()
vad_queue       = queue.Queue()

previous_speech = None
last_lang       = ""
all_texts       = []
html_result     = ""

#
#   General
#
MICROPHONE_SAMPLE_RATE = None
VAD_JOB_RATE           = 0.5

#
#   Recording
#
recordingUtil = RecordingUtil()


#
#   Common voice files
#
d_common_voice_ru = {t[1]: t[2] for t in pd.read_csv('./common_voice_ru/common_voice_ru.csv').values}