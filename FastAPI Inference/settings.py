import torch
import whisper 


def init_globals(static_url, live_url):
    """
    Initialize the global variables of the program.
    :param static_url:
    :param live_url:
    :return:
    """
    global audio_vec, transcribe, transcription, languages, curr_lang, vad, vad_iterator, STOP, FIRST, streaming,\
           STATIC_URL, LIVE_URL, speech_probs, LANGUAGES, get_speech_timestamps, collect_chunks, vad_debug, current_streamming_time

    current_streamming_time = 0
    FIRST = True
    STOP = False
    streaming = True
    curr_lang = ''
    audio_vec = torch.tensor([])
    speech_probs = []
    transcribe = ''
    transcription = ['']
    languages = []
    STATIC_URL = static_url
    LIVE_URL = live_url

    vad, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False,
                                    onnx=False)

    vad_debug, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False,
                                    onnx=False)

    print('loaded silero')
    global get_speech_timestamps, read_audio, save_audio, collect_chunks
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = vad_utils
    vad_iterator = VADIterator(vad)

    LANGUAGES = whisper.tokenizer.LANGUAGES