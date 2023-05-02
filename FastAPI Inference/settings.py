import torch
import whisper
from recording_util import RecordingUtil

def init_globals(static_url, live_url):
    """
    Initialize the global variables of the program.
    :param static_url:
    :param live_url:
    :return:
    """
    global audio_vec, transcribe, transcription, languages, curr_lang, vad, vad_iterator, STOP, FIRST, streaming,\
           STATIC_URL, LIVE_URL, speech_probs, LANGUAGES, get_speech_timestamps, collect_chunks, vad_debug, \
           current_streamming_time, recordingUtil, record_4_debug, num_lang_results, compression_ratio_threshold, logprob_threshold, \
           no_speech_threshold, settings_record_wav, settings_decoding_lang, settings_use_prompt

    settings_record_wav = False
    settings_decoding_lang = []
    settings_use_prompt = False
    recordingUtil = RecordingUtil()
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
    record_4_debug = False
    # the number of detected languages results we need to decide the language. If we have less results, we do not decide the language.
    # when we have enough, we keep the last num_lang_results and report it's mean
    num_lang_results = 5
    # the thresholds below are used to filter out invalid whisper transcriptions
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.6


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