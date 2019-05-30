'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
# import codecs
import re
import os
import unicodedata

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def kor_normalize(text):
    # todo
    return text

def load_data(mode="train"):
    '''Loads training data
      Args:
          mode: "train" or "synthesize" (todo: remove "synthesize")
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode=="train":
        if "LJ" in hp.data:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = open(transcript, 'r').readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                text = text_normalize(text) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
            # fpath is .../LJSpeechx_x/wavs/file.wav
            return fpaths, text_lengths, texts
        elif 'korean' in hp.data: # korean-single-speaker-speech-dataset
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = open(transcript, 'r').readlines()
            for line in lines:
                fname, _, text, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, "wavs", fname.split('/')[-1])
                fpaths.append(fpath)

                text += " E"  # E: EOS
                text = [char2idx[char] for char in text.split(' ')]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
            return fpaths, text_lengths, texts
        else: # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = open(transcript, 'r').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts

    else: # synthesize on unseen test text.
        # Parse
        if 'korean' in hp.test_data:
            lines = open(hp.test_data, 'r').readlines()[1:]
            sents = [kor_normalize(line.split(" ", 1)[-1]).strip() + " E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent.split(' ')]
        else:
            lines = open(hp.test_data, 'r').readlines()[1:]
            sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def load_new_data(filename):
    # Load vocabulary
    char2idx, idx2char = load_vocab()
    # Parse
    lines = open(hp.test_data, 'r').readlines()[1:]
    sents = [kor_normalize(line.split(" ", 1)[-1]).strip() + " E" for line in lines]  # text normalization, E: EOS
    texts = np.zeros((len(sents), hp.max_N), np.int32)
    for i, sent in enumerate(sents):
        texts[i, :len(sent)] = [char2idx[char] for char in sent.split(' ')]
    return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        # LJS fpath is .../LJSpeechx_x/wavs/file.wav
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                try:
                    melp = os.path.join(hp.data, "mels", "{}".format(fname.replace("wav", "npy")))
                    magp = os.path.join(hp.data, "mags", "{}".format(fname.replace("wav", "npy")))
                except TypeError:
                    melp = os.path.join(hp.data, "mels", "{}".format(fname.decode("utf-8").replace("wav", "npy")))
                    magp = os.path.join(hp.data, "mags", "{}".format(fname.decode("utf-8").replace("wav", "npy")))
                mel = np.load(melp)
                mag = np.load(magp)
                return fname, mel, mag

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.B*4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

