import whisper
import numpy as np
import datetime
import requests
import torch
from scipy.io.wavfile import write
import librosa
import settings
import logging



def create_bad_response():
    bad_response = {}
    bad_response["text"]              = "error"
    bad_response["language"]          = "error"
    bad_response["no_speech_prob"]    = 0
    bad_response["avg_logprob"]       = 0
    bad_response["compression_ratio"] = 0

    return bad_response

def Get_Whisper_From_Server(audio_data):

    if settings.RUN_LOCAL_WHISPER is True:
        res = use_local_whisper(audio_data)
        return res

    res        = requests.get(settings.WHISPER_URL, json=audio_data)
    if type(res) == requests.Response:
        if 200 != res.status_code:
            logging.error(f"Got Resonse: {res.status_code} from whisper")
            bad_response = create_bad_response()
            return bad_response

    res =  res.json()[0]
    return res

def use_local_whisper(audio_data):

    audio = [float(i) for i in audio_data["wav"]]
    audio = whisper.pad_or_trim(np.array(audio)).astype('float32')
    if settings.whisper_model is None:
        settings.whisper_model = whisper.load_model("base", device=settings.DEVICE)
    mel   = whisper.log_mel_spectrogram(audio).to(settings.DEVICE)

    # decode the audio
    options         = whisper.DecodingOptions(fp16=False,
                                              task='transcribe',
                                              beam_size  = 5,
                                              language   = audio_data["languages"][0],
                                              prompt     = audio_data["prompt"][0],
                                              sample_len = 100)
    result          = whisper.decode(settings.whisper_model, mel, options)

    whisper_results = {}
    whisper_results['language']          = result.language
    whisper_results["text"]              = result.text
    whisper_results["no_speech_prob"]    = result.no_speech_prob
    whisper_results["avg_logprob"]       = result.avg_logprob
    whisper_results["compression_ratio"] = result.compression_ratio

    return whisper_results



def filter_bad_results(whisper_results):

    if settings.FILTER_BAD_RESULUS is False:
        language = whisper_results["language"]
        text     = whisper_results["text"]
        return text, language


    #
    #   step 1: parse results
    #
    language          = whisper_results['language']
    text              = whisper_results["text"]
    no_speech_prb     = whisper_results["no_speech_prob"]
    avg_logprob       = whisper_results["avg_logprob"]
    compression_ratio = whisper_results["compression_ratio"]


    #
    #   step 2: check thresholds
    #
    res_text = text
    if compression_ratio > settings.compression_ratio_threshold:
        logging.warning(f"ranscription aborted due to compression_ratio ({round(compression_ratio, 2)} > {settings.compression_ratio_threshold}), Language: {language}, Text: {text}")
        res_text = ""
    if avg_logprob < settings.logprob_threshold:
        logging.warning(f"transcription aborted due to avg_logprob: {round(avg_logprob, 2)} < {settings.logprob_threshold}, Language: {language}, Text: {text}")
        res_text = ""
    if no_speech_prb > settings.no_speech_threshold and avg_logprob < -0.6:
        logging.warning(f"transcription aborted due to no_speech_prob: {round(no_speech_prb, 2)} > {settings.no_speech_threshold}, Language: {language}, Text: {text}")
        res_text = ""

    #
    #   step 3: bad expressions
    #
    bad_expressions = \
        ['thanks for watching',
         'thank you for watching',
         'Thank you so much for watching'.lower(),
         'Share this video with your friends on social media'.lower(),
         'MBC 뉴스 이덕영입니다'.lower()]
    for exp in bad_expressions:
        if exp in text.lower():
            logging.warning(f"Skipped (badexpressions), but the text was: {text}")
            res_text = ""
            break

    return res_text, language

if __name__ == "__main__":

    print(settings.DEVICE)



    test_file = "init.wav"
    y, sr     = librosa.load(test_file, sr=16000)

    audio_data = {}
    audio_data["wav"]       = [str(i) for i in y.tolist()]
    audio_data["prompt"]    = ["None"]
    audio_data["languages"] = [None]
    res = Get_Whisper_From_Server(audio_data)
    print(res.text)





