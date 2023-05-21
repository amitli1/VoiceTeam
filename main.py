import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


import pandas               as pd
import utilities
import whisperHandler
DEBUG = True

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


import gradio as gr
import numpy as np
import settings
import torch
import whisper
import librosa
import html_utils
import diarizationHandler
import global_parameters
import matplotlib.pyplot               as plt
import plotly.express                  as px



def schedule_preprocess_speech_job():

    #
    #   Step 1: collect last (5) seconds speech
    #
    q_len = global_parameters.speech_queue.qsize()
    if q_len == 0:
        return
    speech = np.array([])
    for i in range(q_len):
        speech = np.concatenate((speech, global_parameters.speech_queue.get()))

    speech_16000 = librosa.resample(speech, orig_sr=global_parameters.MICROPHONE_SAMPLE_RATE, target_sr=16000)
    speech       = speech_16000

    #
    #   Step 2: check if we have older speech which we didn't finished to preocess
    #
    if global_parameters.previous_speech is not None:
        speech = np.concatenate((global_parameters.previous_speech, speech))
    global_parameters.previous_speech = None

    #
    #   Step 3: run vad
    #
    speech                 = torch.from_numpy(speech)
    speech_timestamps      = global_parameters.speech_get_speech_timestamps(speech.float(),
                                                                              global_parameters.vad_speech_model,
                                                                              sampling_rate=16000)

    #
    #   Step 4: check if we have speech
    #
    if len(speech_timestamps) == 0:
        logging.debug("Add new line (sentence finished)")
        global_parameters.processed_queue.put("\n")
        return

    for val in speech_timestamps:
        start_sample = val['start']
        end_sample   = val['end']
        tmp_speech   = speech[start_sample:end_sample].numpy()
        global_parameters.processed_queue.put(tmp_speech)
        logging.info(f"Push speech to whisper Q, audio length: {round(len(tmp_speech) / 16000, 2)} seconds, |Q| = {global_parameters.processed_queue.qsize()}")


    #
    #   if we need to add new line
    #
    if val['end'] < len(speech) and ((len(speech) - val['end']) >= 16000*0.2):
        logging.debug("Add new line (sentence finished)")
        global_parameters.processed_queue.put("\n")

    if val['end']  >= len(speech):
        global_parameters.previous_speech = speech[speech_timestamps[-1]["start"]:].numpy()


def schedule_vad_job():

    #
    #   if no new mic samples
    #
    vad_q_len = global_parameters.vad_queue.qsize()
    if vad_q_len == 0:
        return global_parameters.last_vad_plot

    #
    #   collect samples (probably ~10 items for first time, and ~2 items for the others)
    #
    speech = np.array([])
    for i in range(vad_q_len):
        speech                                 = np.concatenate((speech, global_parameters.vad_queue.get()))
        global_parameters.total_num_of_vad_elm = global_parameters.total_num_of_vad_elm + 1

    #
    # run VAD (for each half second)
    #
    arr_vad                          = []
    VAD_WINDOW                       = int(global_parameters.MICROPHONE_SAMPLE_RATE * global_parameters.VAD_JOB_RATE)

    for i in range(0, len(speech), VAD_WINDOW):
        chunk = speech[i: i + VAD_WINDOW]
        chunk = torch.from_numpy(chunk)
        chunk = chunk.float()
        if len(chunk) < VAD_WINDOW:
            break
        speech_dict = global_parameters.vad_iterator(chunk, return_seconds=True)
        if speech_dict:
            None
        speech_prob = global_parameters.vad_model(chunk, 16000).item()
        arr_vad.append(speech_prob)

    #
    # add results to last results (and save only last 30 seconds -> last 60 VAD probabilties
    #
    if global_parameters.last_30_sec_vad is None:
        global_parameters.last_30_sec_vad = arr_vad
    else:
        global_parameters.last_30_sec_vad   = global_parameters.last_30_sec_vad + arr_vad
        NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS = int(30/global_parameters.VAD_JOB_RATE)
        if len(global_parameters.last_30_sec_vad) > NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS:
            start_offset = len(global_parameters.last_30_sec_vad) - NUMBER_OF_VAD_PROBABILTIES_IN_30_SECONDS
            global_parameters.last_30_sec_vad = global_parameters.last_30_sec_vad[start_offset:]

    #
    #   create plot
    #
    if len(global_parameters.last_30_sec_vad) < 2:
        return global_parameters.last_vad_plot

    end_time   = global_parameters.total_num_of_vad_elm
    start_time = max(end_time - len(global_parameters.last_30_sec_vad), 0)
    x_time = np.arange(start=start_time, stop=end_time, step=1)
    x_time = x_time / 2


    if len(global_parameters.last_30_sec_vad) != len(x_time):
        logging.error(f"last_30_sec_vad len= {len(global_parameters.last_30_sec_vad)}, x_time len = {len(x_time)}")
        min_val = min(len(global_parameters.last_30_sec_vad), len(x_time))
        x_time                            = x_time[:min_val]
        global_parameters.last_30_sec_vad = global_parameters.last_30_sec_vad[:min_val]

    vad_speech = (np.array(global_parameters.last_30_sec_vad) > 0.5)
    vad_speech = vad_speech.astype(int)



    df           = pd.DataFrame()
    df['time']   = x_time
    df['vad']    = global_parameters.last_30_sec_vad
    df['speech'] = vad_speech
    fig          = px.line(df, x = "time", y="vad", title='silero-vad')
    global_parameters.last_vad_plot = fig

    #
    #   return vad (last 30 seconds) figure
    #
    return fig




def add_new_whisper_results(all_results, text, lang):

    if len(all_results) == 0:
        all_results.append((text, lang))
    else:
        last_text, last_lang = all_results[-1]
        if text == "\n" and last_text == "\n":
            None
        else:
           if text == "\n":
               all_results.append((text, lang))
           else:
               logging.info(f"Replace Last Text: {all_results[-1][0]} With: {text}")
               all_results[-1] = (text, lang)

    # if text == "\n":
    #     logging.info(f"got: NEW LINE, final all_results = {all_results}")
    # else:
    #     logging.info(f"got: {text}, final all_results = {all_results}")
    return all_results


def schedule_whisper_job():

    #
    #   Step 1: get current Q len (number of speech to decode with whisper)
    #
    q_len       = global_parameters.processed_queue.qsize()

    #
    #   step 2: for each speech -> get whisper results
    #
    if q_len != 0:
        for i in range(q_len):
            speech = global_parameters.processed_queue.get()

            if speech == "\n":
                global_parameters.all_texts = add_new_whisper_results(global_parameters.all_texts, "\n", "he")
            else:
                audio_data = {}
                audio_data["wav"]       = [str(ii) for ii in speech.tolist()]
                audio_data["prompt"]    = ["None"]
                audio_data["languages"] = [None]
                whisper_results         = whisperHandler.Get_Whisper_From_Server(audio_data)
                text, language          = whisperHandler.filter_bad_results(whisper_results)
                if text != "":
                    global_parameters.last_lang = language
                    global_parameters.all_texts = add_new_whisper_results(global_parameters.all_texts, text, language)
                    logging.info(f"Got Good Results from Whisper, Text: {text} \tLanguage: {language}, Speech Len: {round(len(speech)/16000, 2)} Seconds")

                if settings.record_to_wav is True:
                    global_parameters.recordingUtil.record_wav(speech, sample_rate=16000)

    #
    #   Step 3: return results
    #
    html_text = html_utils.build_html_table(global_parameters.all_texts)
    html_text = ''.join(html_text)
    return  html_text



def handle_offline_single_speech(audioRecord, audioUpload):

    #
    #   step 1: check if input is valid
    #
    logging.info("User choose to work with offline whisper")
    if (audioRecord is None) and (audioUpload is None):
        res = "<p style='color:red; text-align:left;'> Input is missing </p>"
        return res

    if (audioRecord is not None) and (audioUpload is not None):
        res = "<p style='color:red; text-align:left;'> Two inputs are selected, choose one of them </p>"
        return res

    filePath = audioRecord
    if audioRecord is None:
        filePath = audioUpload


    #
    #   step 2: if we need to use saved results
    #
    file_name = filePath.split('/')[-1][0:-10]
    if f"{file_name}.mp3" in global_parameters.d_common_voice_ru:
        logging.info(f"Work with saved file results: {file_name}")
        text     = global_parameters.d_common_voice_ru[f"{file_name}.mp3"]
        language = "ru"
    else:
        #
        #   step 3: prepare input to whisper server
        #
        logging.info("Prepare wav file to whisper server")
        audio = whisper.load_audio(filePath)
        audio_data = {}
        audio_data["wav"]       = [str(ii) for ii in audio.tolist()]
        audio_data["prompt"]    = ["None"]
        audio_data["languages"] = [None]
        whisper_results = whisperHandler.Get_Whisper_From_Server(audio_data)

        #
        #   step 4: get whisper results
        #
        language  = whisper_results['language']
        text      = whisper_results["text"]

    #
    #   step 5: prepare html
    #
    logging.info("Prepare HTML results")
    all_texts = []
    all_texts = add_new_whisper_results(all_texts, text, language)
    html_text = html_utils.build_html_table(all_texts)
    html_text = ''.join(html_text)
    return html_text





def handle_wav_file(audioRecord, audioUpload):
    '''
    :param input_file:  wav file (Sample rate doesn't matter)
    :return:            speaker diarization plot
                                  +
                        whisper results for each speaker
    '''

    # audioRecord, audioUpload
    if (audioRecord is None) and (audioUpload is None):
        res = "<p style='color:red; text-align:left;'> Input is missing </p>"
        return res

    if (audioRecord is not None) and (audioUpload is not None):
        res = "<p style='color:red; text-align:left;'> Two inputs are selected, choose one of them </p>"
        return res

    if audioRecord is not None:
        input_file = audioRecord
    else:
        input_file = audioUpload

    if settings.run_online_pyyannote is False:
        with open(f"{settings.SAVE_RESULTS_PATH}/html_res.txt", 'r') as file:
            html_whisper_text = file.read()

        diarization_figure, ax = plt.subplots()
        import matplotlib.image as mpimg
        img = mpimg.imread(f"{settings.SAVE_RESULTS_PATH}/fig_res.png")
        ax.imshow(img)
        return html_whisper_text

    if (input_file is None):
        logging.info("Missing WAV file")
        return None


    html_whisper_text = diarizationHandler.run_on_file(input_file)

    return html_whisper_text


def int2float(sound):
    #
    # took from:  https://github.com/snakers4/silero-vad/blob/master/examples/colab_record_example.ipynb
    #
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

def handle_streaming(audio):

    rate  = audio[0]
    voice = audio[1]
    voice = int2float(voice)

    if global_parameters.MICROPHONE_SAMPLE_RATE is None:
        logging.info(f"MICROPHONE_SAMPLE_RATE = {rate}")
        global_parameters.MICROPHONE_SAMPLE_RATE = rate

    if rate != global_parameters.MICROPHONE_SAMPLE_RATE:
        logging.info(f"-sample rate changed: {global_parameters.MICROPHONE_SAMPLE_RATE} ->  {rate} - \n")
        global_parameters.MICROPHONE_SAMPLE_RATE = rate

    global_parameters.vad_queue.put(voice)
    global_parameters.speech_queue.put(voice)





def set_use_language(use_lang):
    logging.info(f"change settings use language: {use_lang}")
    if use_lang == "Hebrew":
        settings.use_language = ["he"]
    elif use_lang == "English":
        settings.use_language = ["en"]


def set_record_to_wav(value):
    logging.info(f"change settings record to wav: {value}")
    settings.record_to_wav = value


def set_use_prompt(value):
    logging.info(f"Change settings, use prompt: {value}")
    settings.user_prompt = value


def use_filter_bad_results(value):
    logging.info(f"Change settings, filter bad results: {value}")
    settings.FILTER_BAD_RESULUS = value

def create_gui():
    with gr.Blocks(theme=gr.themes.Glass()) as demo:

        with gr.Tab("Real Time"):
            stream_input       = gr.Audio(source="microphone")
            output_stream_text = gr.outputs.HTML(label="Whisper Results:")
            output_stream_plt  = gr.Plot(labal = "Voice Activity Detection:")

        with gr.Tab("Whisper Offline"):
            with gr.Row():
                audioUpload = gr.Audio(source="upload", type="filepath")
                audioRecord = gr.Audio(source="microphone", type="filepath")

            audioProcessRecButton = gr.Button("Process")
            output_offline_text   = gr.outputs.HTML(label="")
            audioProcessRecButton.click(fn=handle_offline_single_speech, inputs=[audioRecord, audioUpload],
                                        outputs=[output_offline_text])

        with gr.Tab("Settings"):
            settings_record_wav = gr.Checkbox(label="Record WAV", info="Record WAV files for debug")
            settings_decoding_lang = gr.Dropdown(["None", "Hebrew", "English"], label="DecodingLanguage",
                                                 info="Run Whisper with language decoding")
            settings_use_prompt         = gr.Checkbox(label="Use Whisper prompt", info="Run Whisper with prompt decoding")
            settings_filter_bad_results = gr.Checkbox(label="Filter bad results", info="Filtter whisper bad results")

            settings_record_wav.change(set_record_to_wav,   inputs=[settings_record_wav],    outputs=[])
            settings_decoding_lang.change(set_use_language, inputs=[settings_decoding_lang], outputs=[])
            settings_use_prompt.change(set_use_prompt,      inputs=[settings_use_prompt],    outputs=[])
            settings_filter_bad_results.change(use_filter_bad_results, inputs=[settings_filter_bad_results], outputs=[])

        with gr.Tab("About"):
            gr.Label("Version 2")

        stream_input.stream(fn      = handle_streaming,
                            inputs  = [stream_input],
                            outputs = [])

        demo.load(schedule_vad_job, None, [output_stream_plt], every=global_parameters.VAD_JOB_RATE)
        demo.load(schedule_preprocess_speech_job, None, None, every=2)
        demo.load(schedule_whisper_job, None, [output_stream_text], every=1)

    return demo




if __name__ == "__main__":

    utilities.save_huggingface_token()
    demo = create_gui()
    demo.queue().launch(share=False, debug=False)

    #  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
    #  https://10.53.140.33:8432/

    # demo.queue().launch(share=False,
    #                     debug=False,
    #                     server_name="0.0.0.0",
    #                     server_port=8433,
    #                     ssl_verify=False,
    #                     ssl_certfile="cert.pem",
    #                     ssl_keyfile="key.pem")
