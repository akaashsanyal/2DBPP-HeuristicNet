import sys
import numpy as np
import pandas as pd
import rectpack
from rectpack import newPacker, float2dec
from utils import * 

data_file = ""
dataset = read_dataset(data_file)
for instance in dataset:

    packer = newPacker(mode=PackingMode.Offline, \
                      bin_algo=PackingBin.BBF, \
                      pack_algo=MaxRectsBssf, \
                      sort_algo=SORT_AREA, \
                      rotation=True)
    items = instance.items
    bins = instance.binsize
