from settings import init_globals
from demo_utils import *

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


def gradio_main(static_url, live_url, debug_flag=False):
    init_globals(static_url, live_url)

    block = gr.Blocks(css=css)

    with block:
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
                      Whisper
                    </h1>
                  </div>
                </div>
            """
        )
        with gr.Group():
            plot = gr.Plot(show_label=False, visible=False)
            with gr.Row():
                with gr.Box():
                    radio = gr.Radio(["סטרימינג", "הקלטה", "קובץ"], label="?איך תרצה לספק את האודיו")
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

            text = gr.Textbox(show_label=False, elem_id="result-textarea")
            text2 = gr.Textbox(show_label=False, elem_id="result-textarea")
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
            gr.HTML('''

               <img style="text-align: center; max-width: 650px; margin: 0 auto;"
                src="https://geekflare.com/wp-content/uploads/2022/02/speechrecognitionapi.png",
                width="500" height="600">

            ''')

    block.queue().launch(debug=debug_flag, )


# if __name__ == '__main__':
#     main()
