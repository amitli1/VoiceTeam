import json
import io
import time
import torch
import librosa
import requests
import threading
import gradio as gr
import soundfile as sf
from queue import Queue
from pygame import mixer
from statistics import mode
import plotly.express as px
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
import sys, os
import pandas as pd
import re
import whisper
import numpy as np
import settings
import traceback

def change_audio(audio_type):
    """
    The function allows the user to choose a way to provide audio
    :param string:
    :return:
    """

    if audio_type == 'Streaming':
        settings.streaming = True
        return gr.update(visible=True),  gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False)
    elif audio_type == 'Recording':
        settings.streaming = False
        return gr.update(visible=False),  gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=True), gr.update(visible=True)
    else:
        settings.streaming = False
        return gr.update(visible=False),  gr.update(visible=True), gr.update(visible=True), \
            gr.update(visible=False), gr.update(visible=False)


def play_sound():
    """
    This function is being used when the user wants to play his uploaded file/recording.
    """
    sf.write('uploaded.wav', data=settings.audio_vec, samplerate=16000)
    mixer.init()
    mixer.music.load('uploaded.wav')
    mixer.music.play()

def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def transcribe_chunk_live(audio, prompt = None):
    """
    This functions transcribe given audio chunk. It also determines the language of the chunk and append it to the
    global languages list. It sends the audio to the server.
    :param audio:
    :return: str: transcription
    """

    try:
        # settings.recordingUtil.record_wav(audio)
        num_of_samples_before_vad = len(audio)
        start = time.time()
        speech_timestamps = settings.get_speech_timestamps(audio, settings.vad_debug, sampling_rate=16000)
        if len(speech_timestamps) != 0:
            audio             = settings.collect_chunks(speech_timestamps, audio)
        else:
            print("!!!!!!!!!!!!!!!!")
            print("After VAD there is no speech here, return nothing!")
            print("!!!!!!!!!!!!!!!!")
            return None
        end = time.time()
        num_of_samples_after_vad = len(audio)
        print(f"[transcribe_chunk_live]: VAD2 took {end - start} seconds\n\tBefore VAD: {round(num_of_samples_before_vad/16000, 2)} seconds\n\tAfter VAD: {round(num_of_samples_after_vad/16000, 2)} seconds")
        start = time.time()

        if prompt is None:
            prompt = ['None']
        else:
            prompt = [prompt]
        audio_data = {'wav': [str(i) for i in audio.tolist()], 'languages': settings.settings_decoding_lang, 'prompt':prompt}
        if settings.RUN_LOCAL:
            res = get_local_transcription(audio_data)[0]
        else:
            if audio_data['prompt'] == ['']:
                audio_data['prompt'] = ['None']
            res = requests.get(settings.LIVE_URL, json=audio_data)
            if res.status_code != 200:
                print(f"[ERROR], Got Status code: {res.status_code}")
                settings.recordingUtil.record_wav_for_investigation(audio, must_record=True, json_data=audio_data)
                return None
            tmp = res.json()[0]
            res = tmp
        end = time.time()
        print(f"[transcribe_chunk_live]: transcribe took {end - start} seconds")
    except Exception as e:
        print("\nError in transcribe_chunk_live\n")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"exc_type: {exc_type}, fname: {fname}, line: {exc_tb.tb_lineno}")
        print(f"\naudio type = {type(audio)}\n")
        print(f"\naudio len = {len(audio)}\n")
        print(f"\naudio_data[language] = {audio_data['languages']}")
        print(f"\naudio_data[prompt] = {audio_data['prompt']}")
        print(f"\nspeech_timestamps = {speech_timestamps}")
        settings.recordingUtil.record_wav_for_investigation(audio, must_record=True, json_data=audio_data)
        print("\n")
        return None
    return res


def transcribe_chunk(audio, prompt):
    """
    This functions transcribe given audio chunk. It also determines the language of the chunk and append it to the
    global languages list.
    :param audio:
    :return: str: transcription
    """
    res = transcribe_chunk_live(audio, prompt)
    if res is None:
        return "", ""
    if settings.RUN_LOCAL:
        text = res.text.strip()
        lang = res.language
    else:
        text = res['text'].strip()
        lang = res['language']
    return text, lang



def get_local_transcription(audio_data):
    wav_list  = audio_data["wav"]
    prompt    = audio_data["prompt"]
    languages = audio_data["languages"]

    # if prompt == [''] or prompt == ['None'] or prompt is None:
    #     prompt = ''
    if prompt is None:
        prompt = ''
    if type(prompt) == list:
        if len(prompt) > 0:
            prompt = prompt[0]
    if (languages == ['']) or (languages == ['None']) or (languages is None) or (languages == [None]):
        languages = []

    wav     = [np.float(i) for i in wav_list]
    audio   = whisper.pad_or_trim(np.array(wav)).astype('float32')
    mel     = whisper.log_mel_spectrogram(audio).to('cuda')
    try:
        options = whisper.DecodingOptions(fp16=True, task='transcribe', beam_size=5, prompt = prompt, language=languages)
        result  = whisper.decode(settings.audio_model, mel, options)
    except Exception as e:
        print(f"Got exception: {e}")
        print(f"languages = {languages}")
        print(f"prompt = {prompt}")
        print("\n")
    return [result]

def inference_file(audio):
    """
    This function is the main function. It creates the VAD plot and call one of the transcription functions.
    :param audio:
    :return: current language, vad fig, gr.update(visible=True), transcribe, gr.update(visible=True), gr.update(visible=True)
    """

    # time.sleep(0.2)
    # amitli: when streaming is stopped -> we will clear the MIC thread queue
    if settings.current_streamming_time == 0:
        print("First time get into inference_file function")
    settings.current_streamming_time = datetime.now()

    if settings.FIRST and settings.streaming:
        print('open thread - realtime')
        thread1 = threading.Thread(target=realtime)
        thread1.start()
        settings.STOP = False
        settings.FIRST = False

    wav = torch.from_numpy(librosa.load(audio, sr=16000)[0])
    settings.audio_vec = torch.cat((settings.audio_vec, wav))
    speech_probs = settings.speech_probs

    # j is the start point (in seconds) where the vad graph will start
    j = max(0, len(speech_probs) // 10 - 30)

    window_size_samples = 1600
    x = []
    y = []
    start_time = time.time()
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i + window_size_samples]
        if len(chunk) < window_size_samples:
            break

        speech_prob = settings.vad(chunk, 16000).item()
        speech_probs.append(speech_prob)
    settings.speech_probs = speech_probs
    if len(speech_probs) > 300:
        speech_probs = speech_probs[-300:]
    settings.vad_iterator.reset_states()
    end_time = time.time()
    #print(f"[inference_file]: VAD1 took {end_time - start_time} seconds")

    sample_per_sec = 16000 / window_size_samples
    if not settings.streaming:
        j = max(0, len(wav) // 16000 - 30)
    x.extend([j + i / sample_per_sec for i in range(len(speech_probs))])
    y.extend(speech_probs)

    df = pd.DataFrame()
    df['Time'] = x
    df['Speech Probability'] = y
    fig = px.line(df, x='Time', y='Speech Probability', title='Voice Activity Detection')
    fig.add_scatter(opacity=0, x=[j], y=[1])
    fig.update_layout(showlegend=False)
    # delete if not needed
    wav = settings.audio_vec
    phrases = []
    if not settings.streaming:
        if wav.shape[0] > 16000 * 30:
            start = 0
            end = 16000 * 30
            chunk = wav[start:end]
            chunk_idx = 0
            while end < wav.shape[0]:
                prompt = None
                if settings.settings_use_prompt and len(phrases) > 0:
                    prompt = phrases[-1]
                # temp_trans, temp_langu =
                text, lang = transcribe_chunk(chunk, prompt)
                settings.transcribe += text
                settings.transcription_lang = lang
                phrases.append(text)
                chunk_idx += 1
                start = chunk_idx * 30 * 16000
                if start >= wav.shape[0]:
                    break
                end = (chunk_idx + 1) * 30 * 16000
                if end >= wav.shape[0]:
                    end = wav.shape[0] - 1
                    chunk = wav[start:end]
        else:

            if True:
                # --- get file name (without path and without gradio tmp file extenstions)
                fileName = audio.split("/")[-1][0:-10]

                # --- if we want to get hard coded text
                if fileName + '.mp3' in settings.prefiles_cv10:
                    text = settings.prefiles_cv10[fileName + '.mp3']
                    lang = "ru"
                else:
                    text, lang = transcribe_chunk(wav, prompt = None)
            else:
                text, lang = transcribe_chunk(wav, prompt = None)

            settings.transcribe += text
            settings.transcription_lang = lang

        print(f"detect langs ={settings.languages}")
        if len(settings.languages) > 0:
            settings.curr_lang = mode(settings.languages)

        # add results
        settings.transcription.append(text)


    #text_to_show = show_only_last_rows(settings.transcribe)
    #print(f"---> Total number of whisper results: {len(settings.l_phrases)}, Values: {settings.l_phrases}")
    return settings.curr_lang, fig, gr.update(visible=True),  \
        gr.update(visible=True), gr.update(visible=True)

def build_html_res(l_text, l_lang):
    all_text = []
    all_lang = []

    #
    #   copy just not empty text
    #
    for i in range(len(l_text)):
        if len(l_text[i]) == 0 or l_text[i] == "":
            continue
        all_text.append(l_text[i])
        if len(l_lang) > i:
            all_lang.append(l_lang[i])

    #
    #   save only 20
    #
    if len(all_text) >= 20:
        all_text = all_text[-20:]
        all_lang = all_lang[-20:]

    #
    #   create results
    #
    html_res = []
    for i in range(len(all_text)):
        text = all_text[i]
        lang = None
        if len(l_lang) > i:
            lang = all_lang[i]
        if lang == "he":
            html_res.append(f"<p align='right'> {text} </p>")
        else:
            html_res.append(f"<p> {text} </p>")

    html_table = ''.join(html_res)
    return html_table


def build_html_table(l_text, l_lang):
    all_text = []
    all_lang = []

    #
    #   copy just not empty text
    #
    for i in range(len(l_text)):
        if len(l_text[i]) == 0 or l_text[i] == "":
            continue
        all_text.append(l_text[i])
        all_lang.append(l_lang[i])

    #
    #   save only 20
    #
    if len(all_text) >= 20:
        del all_text[0]
        del all_lang[0]

    #
    #   create table
    #
    html_table = []
    #html_table.append("<table border='1'  align='center' style='width:100%;color:blue;font-size: 24px;'>")
    html_table.append("<table border='1' align='center' style='width:100%'>")
    for i in range(len(all_text)):
        text = all_text[i]
        lang = all_lang[i]
        if lang == "he":
            html_table.append(f"<tr align='right'>")
        else:
            html_table.append(f"<tr align='left'>")
        html_table.append(f"<td> {text}</td>")
        html_table.append("</tr>")
    html_table.append("</table>")

    html_table = ''.join(html_table)
    #print(html_table)
    return html_table


def show_only_last_rows(l_text, num_of_last_lines=20):

    res = []
    l_text = l_text.split("\n")
    if len(l_text) == 0:
        return ""
    for val in reversed(l_text):
        if (val == "\n") or (len(val) == 0) or (val == ""):
            continue
        res.append(val)
        num_of_last_lines = num_of_last_lines - 1
        if num_of_last_lines == 0:
            break

    if len(res) == 0:
        return ""
    if len(res) == 1:
        return res
    res.reverse()
    res = '\n'.join(res)
    return res

def convert_text_to_html(full_html, current_text, l_current_lang):
    '''
        style text results in html format
    '''

    #print(f"full_html = {len(full_html)}, current_text = {len(current_text)}, l_current_lang = {l_current_lang},")
    if len(l_current_lang) == 0:
        return full_html
    #print(f"\tl_current_lang = {l_current_lang[-1]}, {current_text[-1]}")

    full_html = []
    for i in range(len(l_current_lang)):
        lang = l_current_lang[i]
        text = current_text[i]
        if lang == "en":
            current_line = f"<p style='text-align:left; color:green; font-size:32px'> {text} </p>" + "\n"
        else:
            current_line = f"<p style='text-align:right; color:green; font-size:32px'> {text} </p>" + "\n"
        full_html.append(current_line)

    return full_html

def clear():
    """
    The function is being called by the user for cleaning the figure, transcription and the audio file.
    It reset the relevant global variables.
    :return: "Empty" figure and transcription with relevant updates to the Gradio components.
    """
    print('in clear')
    settings.current_streamming_time = 0
    settings.STOP = True
    settings.curr_lang = ''
    settings.audio_vec = torch.tensor([])
    settings.transcribe = ''
    settings.html_transcribe = []
    settings.transcription = ['']
    settings.transcription_lang = []
    settings.l_phrases = []
    settings.languages = []
    settings.speech_probs = []
    settings.FIRST = True
    return '', gr.update(visible=False), gr.update(visible=False), '', gr.update(visible=False),gr.update(visible=False)


def realtime():
    """
    Real time transcription of streaming audio. This function is being called from 'inference_file' and run on a thread.
    It updates the transcription, languages list and the current language.
    """
    print('Start realtime function')

    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()

    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 100  # minimum audio energy to consider for recording
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """

        # Grab the raw bytes and push it into the thread safe queue.
        temp_data = audio.get_raw_data()

        data_queue.put(temp_data)
        #print("Added to the data_queue")

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    print('Start reocording (using speech recognision)')
    # The ``phrase_time_limit`` parameter is the maximum number of seconds that 
    # this will allow a phrase to continue before stopping and returning the part 
    # of the phrase processed before the time limit was reached. 
    # The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, 
    # there will be no phrase time limit.
    phrase_time_limit = 2
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=phrase_time_limit)

    # pause_timeout = 5
    pause_timeout = 3
    # The last time a recording was retrieved from the queue.
    last_phrase_time = None
    while not settings.STOP:
        try:

            # current_time = datetime.now()
            # diff_in_seconds = (current_time - settings.current_streamming_time).seconds
            # if diff_in_seconds >= 1:
            #     print("Cleaning the data_queue!!!!!")
            #     data_queue.queue.clear()

            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                print("++++++++++++++++++++++++++++++++++++")
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if last_phrase_time is not None:
                    print(f"Time passed between recordings: {now - last_phrase_time}")
                if last_phrase_time and now - last_phrase_time > timedelta(seconds=pause_timeout):
                    last_sample = bytes()
                    print("phrase completed! last_sample reset!")
                    phrase_complete = True
                # # This is the last time we received new audio data from the queue.
                # last_phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                print("Got everything from the data queue")
                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_bytes = audio_data.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, _ = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                wav = torch.from_numpy(audio_array)

                prompt = None
                if settings.settings_use_prompt:
                    if phrase_complete:
                        # if the previous phrase is complete then send it as prompt
                        if len(settings.transcription) > 0:
                            prompt = settings.transcription[-1]
                    else:
                        # otherwise, the last phrase is the ongoing phrase so send the one before
                        if len(settings.transcription) > 1:
                            prompt = settings.transcription[-2]

                # call whisper
                result = transcribe_chunk_live(wav, prompt)
                if result == None:
                    text = ""
                    res_lang = ""
                    compression_ratio = 0
                    no_speech_prob = 1
                    avg_logprob = 0
                else:
                    # This is the last time we received new audio data from the queue.
                    last_phrase_time = now
                    
                    if settings.RUN_LOCAL:
                        text = result.text.strip()
                        res_lang = result.language
                        compression_ratio = result.compression_ratio
                        no_speech_prob = result.no_speech_prob
                        avg_logprob = result.avg_logprob
                    else:
                        text = result['text'].strip()
                        res_lang = result['language']
                        compression_ratio = result['compression_ratio']
                        no_speech_prob = result['no_speech_prob']
                        avg_logprob = result['avg_logprob']
                        settings.recordingUtil.record_wav(wav, no_speech_prob)
                    if len(res_lang) > 0:
                        settings.languages.append(settings.LANGUAGES[res_lang])

                    text = filter_bad_results(text, res_lang, compression_ratio, no_speech_prob, avg_logprob)

                    if text != "":
                        if len(settings.languages) > settings.num_lang_results:
                            settings.languages.pop(0)

                    if len(settings.languages) == settings.num_lang_results:
                        settings.curr_lang = mode(settings.languages)

                    # remove all non-text characters, including emojis, but preserve punctuation marks
                    text = re.sub(r'[^\w\s\.,!?\']', '', text)
                    print(f"After filtering, the text is: {text}")
                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # Otherwise, edit the existing one.
                    if phrase_complete:
                        print("phrase_complete")
                        if (text == "") or (len(text) == 0):
                            print("---> [WARN] Got Empty text results")
                        else:
                            settings.transcription.append(text)
                            settings.transcription_lang.append(res_lang)
                            settings.l_phrases.append(text)
                    else:
                        print("replacing the last line with the current text:")
                        settings.transcription[-1] = text

                    # print(f"Full transcription so far:\n{settings.transcription}\n")
                    #print(f"Last transcription :\n{text}\n")

                    if text != '':
                        settings.transcribe = ''
                        for line in settings.transcription:
                            settings.transcribe += line + '\n'

                # Infinite loops are bad for processors, must sleep.
                time.sleep(0.05)
        except Exception as e:
            print(f"\n **************\n Exception:\n {e} \n *********\n")
        except KeyboardInterrupt:
            break
    if settings.STOP:
        stop_listening(wait_for_stop=False)
        print("Stopped the listening&recording thread!")
    print('Out of real time')

def filter_bad_results(text, lang, compression_ratio, no_speech_prob, avg_logprob):
    bad_expressions = \
    ['thanks for watching',
    'thank you for watching',
    'Thank you so much for watching'.lower(),
    'Share this video with your friends on social media'.lower(),
    'MBC 뉴스 이덕영입니다'.lower()]

    should_skip = False
    print(f"\t-->compression_ratio: {round(compression_ratio,2)} , no_speech_prob={round(no_speech_prob,2)}, avg_logprob={round(avg_logprob,2)}")
    

    if compression_ratio > settings.compression_ratio_threshold:
        print(f"\t-->transcription aborted due to compression_ratio ({round(compression_ratio,2)} > {settings.compression_ratio_threshold}), Language: {lang}, Text: {text}")
        should_skip = True
    if avg_logprob < settings.logprob_threshold:
        print(f"\t-->transcription aborted due to avg_logprob: {round(avg_logprob,2)} < {settings.logprob_threshold}, Language: {lang}, Text: {text}")
        should_skip = True
    if no_speech_prob > settings.no_speech_threshold and avg_logprob < -0.5:
        print(f"\t-->transcription aborted due to no_speech_prob: {round(no_speech_prob,2)} > {settings.no_speech_threshold}, Language: {lang}, Text: {text}")
        should_skip = True

    for exp in bad_expressions:
        if exp in text.lower():
            should_skip = True
            break

    if should_skip:
        print(f"Skipped but the text was: {text}")
        return ''
    return text
