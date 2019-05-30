import argparse
import datetime
import json
import numpy as np
import pickle
from playsound import playsound
import requests

parser = argparse.ArgumentParser(description='')
parser.add_argument('-u', '--url', dest='app_url', type=str, default='0.0.0.0',
                    help='host url')
parser.add_argument('-p', '--port', dest='app_port', type=int, default=5000,
                    help='host port')
args = parser.parse_args()
url = "http://{}:{}/api".format(args.app_url, args.app_port)

print("type message to transcribe")
print("[enter] with blank line replays last")

while True:
    text = input("> ")

    if len(text) > 2:

        file = requests.post(url, data=json.dumps({"text": str(text)}))
        with open('r_tmp.wav', 'wb') as wf:
            wf.write(file.content)
        playsound('r_tmp.wav')

    else:
        try:
            playsound('r_tmp.wav')
        except FileNotFoundError:
            continue
