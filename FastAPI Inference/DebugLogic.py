import torch
import librosa
import requests
import numpy as np
from scipy.io.wavfile import write

def remove_not_speech(audio, speech_timestamps, tmp_saved_result):

    if type(audio) == torch.Tensor:
        audio = audio.numpy()

    audio_length = len(audio) / 16000
    print(f"Before VAD: {len(audio)} samples, {audio_length} seconds")
    audio_res    = np.array([])

    for i in range(len(speech_timestamps)):

        # start_time_samples = speech_timestamps[i]['start']
        # end_time_samples   = speech_timestamps[i]['end']

        start_time_samples = max(speech_timestamps[i]['start'] - 10, 0)
        end_time_samples   = min(speech_timestamps[i]['end'] + 10, len(audio))

        current_samples    = audio[start_time_samples : end_time_samples]
        audio_res                = np.concatenate((audio_res, current_samples))

    if tmp_saved_result is not None:
        audio = np.int16(audio / np.max(np.abs(audio)) * 32767) # scale
        write(tmp_saved_result, 16000, audio)


    print(f"After VAD: {len(audio_res)} samples, {len(audio_res)/16000} seconds")
    return torch.from_numpy(audio_res)


def run_vad(audio):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils


    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
    #audio             = remove_not_speech(audio, speech_timestamps, tmp_saved_result="/home/amitli/Downloads/1/tmp.wav")
    audio = collect_chunks(speech_timestamps, audio)
    return audio

def transcribe_chunk(audio):
    """
    This functions transcribe given audio chunk. It also determines the language of the chunk and append it to the
    global languages list.
    :param audio:
    :return: str: transcription
    """

    audio = run_vad(audio)

    STATIC_URL = 'http://10.53.140.33:86/gradio_demo_static/'
    languages = []

    audio_data = {'wav': [str(i) for i in audio.tolist()], 'languages': languages}
    res = requests.get(STATIC_URL, json=audio_data)
    res = res.json()
    trnscrb, languages = res[0], res[1]
    return trnscrb



def test_whisper(file_name):
    audio = torch.from_numpy(librosa.load(file_name, sr=16000)[0])
    STATIC_URL = 'http://10.53.140.33:86/gradio_demo_static/'
    LIVE_URL   = 'http://10.53.140.33:86/gradio_demo_live/'
    languages = ["he"]

    audio_data = {'wav': [str(i) for i in audio.tolist()], 'languages': languages}
    audio_data = {'wav': [str(i) for i in audio.tolist()]}
    res = requests.get(LIVE_URL, json=audio_data)
    res = res.json()
    print(res)


def test_vad_before_whisper():
    # audio = "/home/amitli/Downloads/1/not_ok.wav"
    audio = "/home/amitli/Downloads/1/ok.wav"

    audio_vec = torch.tensor([])
    wav = torch.from_numpy(librosa.load(audio, sr=16000)[0])
    trnscrb = transcribe_chunk(wav)
    print(f"RES = {trnscrb}")


if __name__ == "__main__":

    #test_vad_before_whisper()
    #test_whisper("/home/amitli/Downloads/Voice_Team/2023_5_1_16_2_24/1.wav")
    test_whisper("/home/amitli/Downloads/Voice_Team/2023_5_1_16_36_1/7.wav")


