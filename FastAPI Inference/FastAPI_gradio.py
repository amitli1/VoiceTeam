from settings import init_globals
from demo_utils import *
import gradio as gr
import datetime


def change_settings(settings_record_wav, settings_decoding_lang, settings_use_prompt):
    print(f"Settings changed to: Reocrd Wav: {settings_record_wav}, Decoding Lang: {settings_decoding_lang}, Decoding Prompt:{settings_use_prompt}")
    settings.settings_record_wav = settings_record_wav
    settings.settings_use_prompt = settings_use_prompt
    settings.settings_decoding_lang = None
    if settings_decoding_lang == "Hebrew":
        settings.settings_decoding_lang = "he"
    elif settings_decoding_lang == "English":
        settings.settings_decoding_lang = "en"


def gradio_main(live_url, debug_flag=False, RUN_LOCAL=True):
    init_globals(live_url, RUN_LOCAL)

    # block = gr.Blocks(css=css)
    block = gr.Blocks(theme=gr.themes.Glass())

    with block:
        with gr.Tab("Main"):
            gr.HTML(
                """
                    <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                      <div
                        style="
                          display: inline-flex;
                          align-items: center;
                          gap: 0.8rem;
                          font-size: 1.75rem;
                        "
                      >
                        <h1 style="font-weight: 900; margin-bottom: 7px;">
                          Automatic Speech Recognition
                        </h1>
                      </div>
                    </div>
                """
            )
            with gr.Group():

                with gr.Row():
                    with gr.Box():
                        # radio = gr.Radio(["סטרימינג", "הקלטה", "קובץ"], label="?איך תרצה לספק את האודיו")
                        radio = gr.Radio(["Streaming", "Recording", "Uploaded File"], label="Audio Type:")
                        with gr.Row().style():
                            # Different type of gr.Audio - streaming, recording, uploading
                            audio = gr.Audio(

                                show_label=False,
                                source="microphone",
                                type="filepath",
                                visible=True

                            )
                            audio2 = gr.Audio(

                                label="Input Audio",
                                show_label=False,
                                source="upload",
                                type="filepath",
                                visible=False

                            )
                            audio3 = gr.Audio(
                                label="Input Audio",
                                show_label=False,
                                source="microphone",
                                type="filepath",
                                visible=False
                            )

                            trans_btn = gr.Button("Transcribe", visible=False)
                            trans_btn3 = gr.Button("Transcribe", visible=False)

                text = gr.Textbox(show_label=True, elem_id="result-textarea", label = "Detected Language:")
                text2 = gr.Textbox(show_label=True, elem_id="result-textarea_rtl", label = "Transcription:")

                plot = gr.Plot(show_label=False, visible=False)

                with gr.Row():
                    clear_btn = gr.Button("Clear", visible=False)
                    play_btn = gr.Button('Play audio', visible=False)

                radio.change(fn=change_audio, inputs=radio, outputs=[audio, trans_btn, audio2, trans_btn3, audio3])

                trans_btn.click(inference_file, audio2, [text, plot, plot, text2, clear_btn, play_btn])
                trans_btn3.click(inference_file, audio3, [text, plot, plot, text2, clear_btn, play_btn])
                audio.stream(inference_file, [audio], [text, plot, plot, text2, clear_btn, play_btn])

                play_btn.click(play_sound)
                clear_btn.click(clear, inputs=[], outputs=[text, plot, plot, text2, clear_btn, play_btn])

                gr.HTML('''
                <div class="footer">
                            <p>Model by Moses team - Whisper Demo
                            </p>
                </div>
                ''')
                # gr.HTML('''

                #    <img style="text-align: center; max-width: 650px; margin: 0 auto;"
                #     src="https://geekflare.com/wp-content/uploads/2022/02/speechrecognitionapi.png",
                #     width="500" height="600">

                # ''')
        with gr.Tab("Settings"):
            settings_record_wav    = gr.Checkbox(label="Record WAV", info="Record WAV files for debug")
            settings_decoding_lang = gr.Dropdown(["None", "Hebrew", "English"], label="DecodingLanguage", info="Run Whisper with language decoding")
            settings_use_prompt    = gr.Checkbox(label="Use Whisper prompt", info="Run Whisper with prompt decoding")

            settings_record_wav.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt], outputs=[])

            settings_decoding_lang.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt], outputs=[])

            settings_use_prompt.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt], outputs=[])

    # block.queue().launch(debug=debug_flag, )
#for debugging on vitaly's computer
    block.queue().launch(share=False,
                        debug=debug_flag,
                        server_name="0.0.0.0",
                        server_port=8432,
                        ssl_verify=False,
                        ssl_certfile="cert.pem",
                        ssl_keyfile="key.pem")

# if __name__ == '__main__':
#     main()
