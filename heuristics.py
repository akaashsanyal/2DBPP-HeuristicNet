# Standard imports
import sys
import random
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
    
    num_heuristics = len(bin_algos) * len(pack_algos)
    labels = []
    # Print progress
    count = 1
    # Repeat for each heuristic
    for instance in dataset:
        instance_label = []
        for bin_algo in bin_algos:
            for pack_algo in pack_algos:
                # Initialize Packer
                packer = newPacker(mode=PackingMode.Offline, \
                                bin_algo=bin_algo, \
                                pack_algo=pack_algo, \
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
                
                instance_label.append(len(packer))
                
        # Save results
        indices = [i for i, x in enumerate(instance_label) if x == min(instance_label)]
        # In a tie, randomly pick one
        correct = random.choice(indices)
        labels.append(correct)
        if count%10 == 0:
            print("Done with instance " + str(count))
        count += 1
    one_hot = np.zeros((len(labels), num_heuristics))
    one_hot[np.arange(len(labels)),labels] = 1
    return one_hot, num_heuristics

