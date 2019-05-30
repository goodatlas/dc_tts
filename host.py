'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import argparse
import os
import re
import librosa
import jamotools
from playsound import playsound
import datetime
import json

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_new_data
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm
from flask import Flask, request, render_template, redirect, url_for, session, make_response, send_file

class TTS:
    def __init__(self):
        # Load graph
        self.char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
        self.g = Graph(mode="synthesize")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        self.saver1 = tf.train.Saver(var_list=var_list)
        model1 = tf.train.latest_checkpoint(hp.logdir + "-1")
        self.saver1.restore(self.sess, model1)
        print("LOADED: Text2Mel Restored from {}".format(model1))
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        self.saver2 = tf.train.Saver(var_list=var_list)
        model2 = tf.train.latest_checkpoint(hp.logdir + "-2")
        self.saver2.restore(self.sess, model2)
        print("LOADED: SSRN     Restored from {}".format(model2))


    def _preprocess_korean(self, sent, null='ⅇ'):
        sent = sent.lower()
        sent = re.sub(r'[^가-힣\s\.\,\?\!]', '', sent)
        seq = []
        for c in list(sent):
            if re.match(r'[가-힣]', c):
                jamos = list(jamotools.split_syllables(c))  # use this for positionless
                if jamos[0] == 'ㅇ':  # the 'positionless' nieung
                    jamos = [null] + jamos[1:]
                # print(jamos)
                seq += jamos
            else:
                if c == ' ':
                    c = '▁'
                seq.append(c)
        texts = np.zeros((1, hp.max_N), np.int32)
        texts[0, :len(seq)] = [self.char2idx[char] for char in seq]
        return texts

    def generate(self, sent):
        print(datetime.datetime.now().isoformat()[:19], ": received sentence")
        L = self._preprocess_korean(sent)
        # Feed Forward
        ## mel
        Y = np.zeros((1, hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((1,), np.int32)
        for j in range(hp.max_T):
            _gs, _Y, _max_attentions, _alignments = \
                self.sess.run([self.g.global_step, self.g.Y, self.g.max_attentions, self.g.alignments],
                              {self.g.L: L,
                               self.g.mels: Y,
                               self.g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = self.sess.run(self.g.Z, {self.g.Y: Y})
        print(datetime.datetime.now().isoformat()[:19], ": model decoding done")
        wav = spectrogram2wav(Z[0])
        print(datetime.datetime.now().isoformat()[:19], ": wav generation done")
        wav, _ = librosa.effects.trim(wav)
        print(datetime.datetime.now().isoformat()[:19], ": wav trimming done")
        return wav

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def apiquery():
    """api interface"""
    msg = str(json.loads(request.data).get('text', ''))
    wav = tts.generate(msg)
    write_wav('tmp.wav', hp.sr, wav)
    return send_file('tmp.wav', attachment_filename='tmp.wav')

if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=-1, help='specify GPU; default none (-1)')
    parser.add_argument('-u', '--url', dest='app_url', type=str, default='0.0.0.0',
                        help='host url')
    parser.add_argument('-p', '--port', dest='app_port', type=int, default=5000,
                        help='host port')
    args = parser.parse_args()

    # restrict GPU usage here, if using multi-gpu
    if args.gpu >= 0:
        print("restricting GPU usage to gpu/", args.gpu, "\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print("restricting to CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    tts = TTS()

    # test = tts.generate('테스트입니다.')
    # write_wav('tmp.wav', hp.sr, test)
    # playsound('tmp.wav')

    app.run(host=args.app_url, port=args.app_port, use_reloader=False, debug=True)


