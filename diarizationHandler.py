import utilities
import datetime
import settings
import librosa
import whisperHandler
import logging

import matplotlib.pyplot               as plt

from pyannote.core                     import notebook
from tqdm                              import tqdm

def process_diarizartion_results(diarization, speech):
    l_speakers_samples = []
    l_text = []
    l_speaker = []
    language = None

    for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True)):
        start_time = turn.start
        end_time = turn.end
        duration = end_time - start_time
        if duration < 0.1:
            continue

        start_sample                   = int(start_time * settings.SAMPLE_RATE)
        end_sample                     = int(end_time * settings.SAMPLE_RATE)
        speaker_samples                = speech[start_sample:end_sample]
        text, language, no_speech_prob = whisperHandler.Get_Whisper_From_Server(speaker_samples)
        l_speakers_samples.append(speaker_samples)
        l_speaker.append(speaker)
        l_text.append(text)

    return l_speakers_samples, l_speaker, l_text, language


def prepare_text(l_text, l_speaker, language):
    '''

    :param l_text:         list of whisper text
    :param l_speaker:      list of speakers
    :param language:       language (he/en/...)
    :return:               HTML page with language alignment.
                           each speaker with different color
    '''

    if language == "he":
        align = "right"
    else:
        align = "left"

    text_results = ""
    speaker_dict = {}
    colors = ["red",  "blue", "green"]
    for i, sp in enumerate(set(l_speaker)):
        speaker_dict[sp] = colors[i]

    for i in range(len(l_speaker)):
        current_text = f"<p style='color:{speaker_dict[l_speaker[i]]}; text-align:{align};'> {l_speaker[i]} {l_text[i]} </p>" + "\n"
        text_results = text_results + current_text
    return text_results



def save_results_to_file(html_res, fig_res):
    with open(f"{settings.SAVE_RESULTS_PATH}/html_res.txt", 'w') as file:
        file.write(html_res)
    fig_res.savefig(f"{settings.SAVE_RESULTS_PATH}/fig_res.png")



def run_on_file(input_file):

    #
    #   Read file properties
    #
    file_duration = utilities.get_wav_duration(input_file)
    sample_rate = utilities.get_sample_rate(input_file)

    if sample_rate != 16000:
        logging.info("Change sample rate to 16000")
        utilities.convert_to_16sr_file(input_file, input_file)

    #
    # run diarization
    #
    logging.info(f"Run Diarization pipeline ({datetime.now()})")
    diarization_res = settings.sd_pipeline(input_file)
    speech, sr = librosa.load(input_file, sr=settings.SAMPLE_RATE)
    res = process_diarizartion_results(diarization_res, speech)
    logging.info(f"process diarzation finished ({datetime.now()})")
    l_speakers_samples = res[0]
    l_speaker          = res[1]
    l_text             = res[2]
    language           = res[3]

    #
    #   create plots
    #
    html_whisper_text = prepare_text(l_text, l_speaker, language)
    diarization_figure, ax = plt.subplots()
    res = notebook.plot_annotation(diarization_res, ax=ax, time=True, legend=True)
    save_results_to_file(html_whisper_text, diarization_figure)

    #
    #   return results
    #
    return html_whisper_text
