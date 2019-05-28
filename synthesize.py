'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import argparse
import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_new_data
from scipy.io.wavfile import write
from tqdm import tqdm

def synthesize(filename, outdir):
    # Load data
    L = load_new_data(filename)

    # Load graph
    g = Graph(mode="synthesize")
    print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir):
            os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)

if __name__ == '__main__':
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=-1, help='specify GPU; default none (-1)')
    parser.add_argument('-f', '--file',  dest='sentences', type=str, default=hp.test_data, help='test_data, def from hp')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default=hp.sampledir, help='sampledir, def from hp')
    args = parser.parse_args()
    # restrict GPU usage here, if using multi-gpu
    if args.gpu >= 0:
        print("restricting GPU usage to gpu/", args.gpu, "\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print("restricting to CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    synthesize(args.sentences, args.outdir)
    print("Done")


