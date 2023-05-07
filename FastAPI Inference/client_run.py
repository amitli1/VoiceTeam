from FastAPI_gradio import gradio_main
import settings
from settings import init_globals

def client_main():
    
    RUN_LOCAL = True
    # TODO - Change debug_flag as you wish. Default of gradio_main is False. Change The urls if needed
    debug_flag = True
    gradio_main(debug_flag, run_local=RUN_LOCAL)
    


if __name__ == '__main__':
    client_main()
