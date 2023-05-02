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
import pandas as pd
import re

import settings


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



def transcribe_chunk_live(audio):
    """
    This functions transcribe given audio chunk. It also determines the language of the chunk and append it to the
    global languages list. It sends the audio to the server.
    :param audio:
    :return: str: transcription
    """

    settings.recordingUtil.record_wav(audio)

    num_of_samples_before_vad = len(audio)
    speech_timestamps = settings.get_speech_timestamps(audio, settings.vad_debug, sampling_rate=16000)
    if len(speech_timestamps) != 0:
        audio             = settings.collect_chunks(speech_timestamps, audio)
    num_of_samples_after_vad = len(audio)
    print(f"[transcribe_chunk_live]\n\tBefore VAD: {num_of_samples_before_vad} samples, {round(num_of_samples_before_vad/16000, 2)} seconds\n\tAfter VAD: {num_of_samples_after_vad} samples, {round(num_of_samples_after_vad/16000, 2)} seconds")

    audio_data = {'wav': [str(i) for i in audio.tolist()]}
    res = requests.get(settings.LIVE_URL, json=audio_data)

    return res.json()[0]


def transcribe_chunk(audio):
    """
    This functions transcribe given audio chunk. It also determines the language of the chunk and append it to the
    global languages list.
    :param audio:
    :return: str: transcription
    """
    audio_data = {'wav': [str(i) for i in audio.tolist()], 'languages': settings.languages}
    res = requests.get(settings.STATIC_URL, json=audio_data)
    res = res.json()
    trnscrb, settings.languages = res[0], res[1]
    return trnscrb


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

    if not settings.streaming:
        if wav.shape[0] > 16000 * 30:
            start = 0
            end = 16000 * 30
            chunk = wav[start:end]
            chunk_idx = 0
            while end < wav.shape[0]:
                # temp_trans, temp_langu =
                settings.transcribe += transcribe_chunk(chunk)
                # settings.languages = temp_langu
                chunk_idx += 1
                start = chunk_idx * 30 * 16000
                if start >= wav.shape[0]:
                    break
                end = (chunk_idx + 1) * 30 * 16000
                if end >= wav.shape[0]:
                    end = wav.shape[0] - 1
                    chunk = wav[start:end]
        else:
            settings.transcribe += transcribe_chunk(wav)
        print(f"detect langs ={settings.languages}")
        if len(settings.languages) > 0:
            settings.curr_lang = mode(settings.languages)

    #html_text = prepaare_text(settings.transcribe)
    return settings.curr_lang, fig, gr.update(visible=True), settings.transcribe, \
           gr.update(visible=True), gr.update(visible=True)


def prepaare_text(text):
    '''
        style text results in html format
    '''
    html_text = ""
    reg = re.compile(r'[a-zA-Z]')
    for i in range(len(text)):
        if reg.match(text[i]):
            current_line = f"<p style='text-align:right;'> {text[i]} </p>"
        else:
            current_line = f"<p style='text-align:left;'> {text[i]} </p>"
        html_res = html_res + current_line + "\n"
    return html_text


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
    settings.transcription = ['']
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
    energy_threshold = 300
    record_timeout = 2
    phrase_timeout = 3


    temp_file = NamedTemporaryFile().name

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()

    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """

        # Grab the raw bytes and push it into the thread safe queue.
        temp_data = audio.get_raw_data()

        data_queue.put(temp_data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    print('Start reocording (using speech recognistion)')
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    while not settings.STOP:
        try:

            current_time = datetime.now()
            diff_in_seconds = (current_time - settings.current_streamming_time).seconds
            if diff_in_seconds >= 1:
                data_queue.queue.clear()

            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                wav = torch.from_numpy(librosa.load(temp_file, sr=16000)[0])

                # call whisper
                result = transcribe_chunk_live(wav)
                text = result['text'].strip()
                print(f"Got Whisper Results: {text}, no_speech_prob = {result['no_speech_prob']}, language = {result['language']}")
                settings.languages.append(settings.LANGUAGES[result['language']])
                # if result['segments']:
                if result['no_speech_prob'] > 0.75:
                    print(f"No speech prob is too high ({result['no_speech_prob']}) => text will be empty")
                    text = ''
                if len(settings.languages) > 3:
                    settings.languages.pop(0)

                settings.curr_lang = mode(settings.languages)

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise, edit the existing one.
                if phrase_complete:
                    settings.transcription.append(text)
                else:
                    settings.transcription[-1] = text
                #print(f"Full transcription so far:\n{settings.transcription}\n")

                if text != '':
                    settings.transcribe = ''
                    for line in settings.transcription:
                        settings.transcribe += line + '\n'


                # Infinite loops are bad for processors, must sleep.
                time.sleep(0.25)

        except KeyboardInterrupt:
            break
    print('Out of real time')
