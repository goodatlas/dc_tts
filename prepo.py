'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

# from __future__ import print_function

from utils import load_spectrograms
from hyperparams import Hyperparams as hp
import os
from data_load import load_data
import numpy as np
import tqdm

# Load data
fpaths, _, _ = load_data() # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists(os.path.join(hp.data, "mels")):
        os.mkdir(os.path.join(hp.data, "mels"))
    if not os.path.exists(os.path.join(hp.data, "mags")):
        os.mkdir(os.path.join(hp.data, "mags"))

    np.save(os.path.join(hp.data, "mels", "{}".format(fname.replace("wav", "npy"))), mel)
    np.save(os.path.join(hp.data, "mags", "{}".format(fname.replace("wav", "npy"))), mag)
