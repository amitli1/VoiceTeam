import torch
import whisper
import os
from recording_util import RecordingUtil



def init_globals(run_local=True):
    """
    Initialize the global variables of the program.
    :param static_url:
    :param live_url:
    :return:
    """
    global audio_vec, transcribe, transcription, languages, curr_lang, vad, vad_iterator, STOP, FIRST, streaming,\
           STATIC_URL, LIVE_URL, speech_probs, LANGUAGES, get_speech_timestamps, collect_chunks, vad_debug, \
           current_streamming_time, recordingUtil, num_lang_results, compression_ratio_threshold, logprob_threshold, \
           no_speech_threshold, settings_record_wav, settings_decoding_lang, settings_use_prompt, RUN_LOCAL, audio_model, html_transcribe,  transcription_lang, \
            settings_record_for_inverstigation, settings_record_text_to_file, l_phrases


    settings_record_wav = False
    settings_decoding_lang = [None]
    settings_use_prompt = False
    settings_record_for_inverstigation = False
    settings_record_text_to_file = False
    recordingUtil = RecordingUtil()
    current_streamming_time = 0
    FIRST = True
    STOP = False
    streaming = True
    curr_lang = ''
    audio_vec = torch.tensor([])
    speech_probs = []
    transcribe = ''
    html_transcribe = []
    transcription = ['']
    transcription_lang = []
    l_phrases = []
    languages = []
    #STATIC_URL = static_url
    LIVE_URL = 'http://10.53.140.33:86/gradio_demo_live/'
    # the number of detected languages results we need to decide the language. If we have less results, we do not decide the language.
    # when we have enough, we keep the last num_lang_results and report it's mean
    num_lang_results = 5
    # the thresholds below are used to filter out invalid whisper transcriptions
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.8
    RUN_LOCAL = run_local
    if RUN_LOCAL:
        audio_model = whisper.load_model('large', 'cuda')
        #audio_model = whisper.load_model('base', 'cuda')

    if RUN_LOCAL:
        loacl_cashe = f"{os.path.expanduser('~')}/.cache/torch/hub/snakers4_silero-vad_master"
        print(f"Run silero-vad from loacl_cashe")
        vad_path = loacl_cashe
        source = "local"
    else:
        print(f"Run silero-vad from internet (torch hub)")
        internet_path = "snakers4/silero-vad"
        vad_path = internet_path
        source = "github"

    vad, vad_utils = torch.hub.load(repo_or_dir=vad_path,
                                    model='silero_vad',
                                    source = source,
                                    force_reload=False,
                                    onnx=False)

    vad_debug, vad_utils = torch.hub.load(repo_or_dir=vad_path,
                                    model='silero_vad',
                                    source = source,
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