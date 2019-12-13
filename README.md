A gossip chatbot based on GPT2, model pretrained from microsoft: https://github.com/microsoft/DialoGPT

Usage:
    ```
        interact.py [-h] [-m MODEL] [-c CONFIG] [-p PORT] [--device DEVICE] [--backend BACKEND]

            optional arguments:
              -h, --help            show this help message and exit
              -m MODEL, --model MODEL
                                    pretrained model path
              -c CONFIG, --config CONFIG
                                    model config path
              -p PORT, --port PORT  listen port, default is 5000
              --device DEVICE       choose to use cpu or cuda:x, default is cuda:0
              --backend BACKEND     choose for backend from: shell, restful, default is shell
    ```

Dockerfile not tested, should be working
