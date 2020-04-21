# Standard imports
import sys
import numpy as np
import pandas as pd

# Rectpack imports
from rectpack import newPacker, float2dec
from rectpack import PackingBin, PackingMode
from rectpack import MaxRectsBl, MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf
from rectpack import SORT_RATIO

def generate_labels(dataset):
    bin_algos = [PackingBin.BNF, PackingBin.BFF, PackingBin.BBF]
    pack_algos = [MaxRectsBl, MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf]
    
    labels = []
    # Repeat for each heuristic
    for instance in dataset:
        instance_label = []
        for bin_algo in bin_algos:
            for pack_algo in pack_algos:
                # Initialize Packer
                packer = newPacker(mode=PackingMode.Offline, \
                                bin_algo=PackingBin.BBF, \
                                pack_algo=MaxRectsBssf, \
                                sort_algo=SORT_RATIO, \
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

                # Evaluate performance
                # Count number of bins
                instance_label.append(1)
                
                
        # Save results
        # something like labels.append(list of heuristic rankings, or just a number)
        labels.append(instance_label)
    return labels

