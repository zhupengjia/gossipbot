#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Gossip Chat via GPT2')
parser.add_argument('-m', '--model', dest="model", help="pretrained model path")
parser.add_argument('-c', '--config', dest="config", help="model config path")
parser.add_argument('-p', '--port', dest='port', default=5000, help="listen port, default is 5000")
parser.add_argument('--device', dest="device", default="cuda:0", help="choose to use cpu or cuda:x, default is cuda:0")
parser.add_argument('--backend', dest='backend', default='shell', help="choose for backend from: shell, restful, default is shell")
args = parser.parse_args()

from gossipbot.backend import Backend

s = Backend(backend_type=args.backend,
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            port=args.port)
s.run()
