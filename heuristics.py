# Standard imports
import sys
import numpy as np
import pandas as pd

# Rectpack imports
from rectpack import newPacker, float2dec
from rectpack import PackingBin, PackingMode
from rectpack import MaxRectsBl, MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf
from rectpack import SkylineMwf, SkylineMwfl, SkylineBl, \
    SkylineBlWm, SkylineMwfWm, SkylineMwflWm
from rectpack import SORT_AREA, SORT_PERI, SORT_DIFF, SORT_SSIDE, \
    SORT_LSIDE, SORT_RATIO, SORT_NONE

def generate_labels(dataset):
    bin_algos = [PackingBin.Global, PackingBin.BNF, PackingBin.BFF, \
                    PackingBin.BBF]
    pack_algos = [MaxRectsBl, MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf, \
                    SkylineMwf, SkylineMwfl, SkylineBl, \
                    SkylineBlWm, SkylineMwfWm, SkylineMwflWm]
    sort_algos = [SORT_AREA, SORT_PERI, SORT_DIFF, SORT_SSIDE, \
                    SORT_LSIDE, SORT_RATIO, SORT_NONE]
    labels = []
    # Repeat for each heuristic
    for instance in dataset:
        # Initialize Packer
        packer = newPacker(mode=PackingMode.Offline, \
                        bin_algo=PackingBin.BBF, \
                        pack_algo=MaxRectsBssf, \
                        sort_algo=SORT_AREA, \
                        rotation=True)

        # Add bins and items to Packer
        items = instance.items
        bins = instance.binsize
        for i in items:
            packer.add_rect(*i)
        packer.add_bin(bins[0], bins[1], count=float("inf"))

        # Start packing
        packer.pack()

        # Full item list
        all_rects = packer.rect_list()
    '''    
        # Evaluate performance   
        print("INSTANCE RESULTS")
        for rect in all_rects:
            print(rect)
    '''
        # Save results
        # something like labels.append(list of heuristic rankings, or just a number)
    
    return labels

