from FastAPI_gradio import gradio_main
import settings
from settings import init_globals

def client_main():
    # TODO - Change debug_flag as you wish. Default of gradio_main is False. Change The urls if needed
    RUN_LOCAL = True
    if RUN_LOCAL:
        # static_url = 'http://127.0.0.1:5000:5000/gradio_demo_static/'
        live_url = ''
    else:
        # static_url = 'http://10.53.140.33:86/gradio_demo_static/'
        live_url = 'http://10.53.140.33:86/gradio_demo_live/'
    debug_flag = True
    gradio_main(live_url, debug_flag, RUN_LOCAL)
    


if __name__ == '__main__':
    client_main()
