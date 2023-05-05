from settings import init_globals
from demo_utils import *
import gradio as gr
import datetime

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }

        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #result-textarea_rtl
        {
         direction: rtl;
        }
        #share-btn-container {
            display: flex; margin-top: 1.5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem
            !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px
            !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
"""

def change_settings(settings_record_wav, settings_decoding_lang, settings_use_prompt, settings_record_errors):
    print(f"Settings changed to: Reocrd Wav: {settings_record_wav}, "
          f"Decoding Lang: {settings_decoding_lang}, "
          f"Decoding Prompt:{settings_use_prompt}, "
          f"settings_record_errors: {settings_record_errors}")
    settings.settings_record_wav = settings_record_wav
    settings.settings_use_prompt = settings_use_prompt
    settings.settings_record_errors = settings_record_errors
    settings.settings_decoding_lang = [None]
    if settings_decoding_lang == "Hebrew":
        settings.settings_decoding_lang = ["he"]
    elif settings_decoding_lang == "English":
        settings.settings_decoding_lang = ["en"]

def update_text_whisper_display_results():

    text_to_show = build_html_table(settings.l_phrases, settings.transcription_lang)
    return text_to_show


def gradio_main(debug_flag=False, run_local=True):
    init_globals(run_local)

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
                textTranscription = gr.Textbox(show_label=True, elem_id="result-textarea_rtl", label = "Transcription:")
                #textTranscription = gr.TextArea(show_label=True, elem_id="result-textarea_rtl", label="Transcription:")
                #textTranscription = gr.outputs.HTML()

                plot = gr.Plot(show_label=False, visible=False)

                with gr.Row():
                    clear_btn = gr.Button("Clear", visible=False)
                    play_btn = gr.Button('Play audio', visible=False)

                radio.change(fn=change_audio, inputs=radio, outputs=[audio, trans_btn, audio2, trans_btn3, audio3])

                trans_btn.click(inference_file, audio2, [text, plot, plot, textTranscription, clear_btn, play_btn])
                trans_btn3.click(inference_file, audio3, [text, plot, plot, textTranscription, clear_btn, play_btn])
                audio.stream(inference_file, [audio], [text, plot, plot, textTranscription, clear_btn, play_btn])
                # trans_btn.click(inference_file, audio2, [text, plot, plot,  clear_btn, play_btn])
                # trans_btn3.click(inference_file, audio3, [text, plot, plot,  clear_btn, play_btn])
                # audio.stream(inference_file, [audio], [text, plot, plot,  clear_btn, play_btn])

                play_btn.click(play_sound)
                clear_btn.click(clear, inputs=[], outputs=[text, plot, plot, textTranscription, clear_btn, play_btn])

                gr.HTML('''
                <div class="footer">
                            <p>Model by Audio team - Whisper Demo
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
            settings_record_errors = gr.Checkbox(label="Record WAV On errors", info="Record for later debug")

            settings_record_wav.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt,
                                                         settings_record_errors], outputs=[])

            settings_decoding_lang.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt,
                                                         settings_record_errors], outputs=[])

            settings_use_prompt.change(change_settings, inputs=[settings_record_wav,
                                                         settings_decoding_lang,
                                                         settings_use_prompt,
                                                        settings_record_errors], outputs=[])

            settings_record_errors.change(change_settings, inputs=[settings_record_wav,
                                                                settings_decoding_lang,
                                                                settings_use_prompt,
                                                                settings_record_errors], outputs=[])

        with gr.Tab("Version"):
            gr.Label("Version 1.0")

        #block.load(update_text_whisper_display_results, None, [textTranscription], every=1)

    block.queue().launch(debug=debug_flag, )
#for debugging on vitaly's computer
    # block.queue().launch(share=False,
    #                     debug=debug_flag,
    #                     server_name="0.0.0.0",
    #                     server_port=8432,
    #                     ssl_verify=False,
    #                     ssl_certfile="cert.pem",
    #                     ssl_keyfile="key.pem")

# if __name__ == '__main__':
#     main()
