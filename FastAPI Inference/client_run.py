from FastAPI_gradio import gradio_main


def client_main():
    # TODO - Change debug_flag as you wish. Default of gradio_main is False. Change The urls if needed
    static_url = 'http://10.53.140.33:86/gradio_demo_static/'
    live_url = 'http://10.53.140.33:86/gradio_demo_live/'
    debug_flag = True

    gradio_main(static_url, live_url, debug_flag)


if __name__ == '__main__':
    client_main()
