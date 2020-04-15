import sys
import numpy as np
import pandas as pd
import rectpack
from rectpack import newPacker, float2dec
from utils import * 

bin_algos = [PackingBin.Global, PackingBin.BNF, PackingBin.BFF, \
                PackingBin.BBF]
pack_algos = [MaxRectsBl, MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf, \
                SkylineMwf, SkylineMwfl, SkylineBl, \
                SkylineBlWm, SkylineMwfWm, SkylineMwflWm]
sort_algos = [SORT_AREA, SORT_PERI, SORT_DIFF, SORT_SSIDE, \
                SORT_LSIDE, SORT_RATIO, SORT_NONE]

data_file = ""
dataset = read_dataset(data_file)

# Repeat for each heuristic
for instance in dataset:

    # Initialize Packer
    packer = newPacker(bin_algo=PackingBin.BBF, \
                      pack_algo=MaxRectsBssf, \
                      sort_algo=SORT_AREA, \
                      rotation=True)
    
    # Add bins and items to Packer
    items = instance.items
    bins = instance.binsize
    for i in items:
        packer.add_rect(i)
    packer.add_bin(bins[0], bins[1], count=float("inf"))

    # Start packing
    packer.pack()

    # Full item list
    all_rects = packer.rect_list()

    # Evaluate performance

    # Save results
