import sys
import pickle
from tqdm import tqdm

from rectpack import newPacker, float2dec
from rectpack import PackingBin, PackingMode
from rectpack import MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf
from rectpack import SORT_NONE, SORT_AREA, SORT_PERI, SORT_DIFF, SORT_SSIDE, SORT_LSIDE, SORT_RATIO

def generate_labels(dataset, save):
    bin_algos = [PackingBin.BNF, PackingBin.BFF, PackingBin.BBF]
    pack_algos = [MaxRectsBssf, MaxRectsBaf, MaxRectsBlsf]
    sort_algos = [SORT_NONE, SORT_AREA, SORT_PERI, SORT_DIFF, SORT_SSIDE, SORT_LSIDE, SORT_RATIO]

    num_heuristics = len(bin_algos) * len(pack_algos)
    labels = []
    # Repeat for each heuristic
    pbar = tqdm(dataset)
    for instance in pbar:
        pbar.set_description("Generating Heuristic Labels")
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
        labels.append(instance_label)
    
    pickle.dump([labels, num_heuristics], open(save, 'wb'))
