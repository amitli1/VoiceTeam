from FastAPI_server import fastapi_main


def server_main():
    host = '0.0.0.0'
    port = 5000

    # Size and device are default as 'medium' and 'cuda:0'. If different settings required,
    # uncomment the next rows and change as you wish.

    # size = 'large'
    # device = 'cpu'
    fastapi_main(host, port)


if __name__ == '__main__':
    server_main()
