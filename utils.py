import sys
import os
import numpy as np

# Instance class
class Instance:
    def __init__(self, binsize=None, items=None):
        self.binsize = binsize or []
        self.items = items or []

    def add_item(self, item):
        self.items.append(item)

    def set_binsize(self, binsize):
        self.binsize = binsize

    def print_instance(self):
        print(self.binsize)
        for item in self.items:
            print(item)

# Make dataset and save to file
def generate_raw_dataset(filepath):
    '''
    Format is as follows:
    instance number
    bin_length bin_width
    item_length item_width item_id
    item_length item_width item_id
    ...
    item_length item_width item_id

    bin_length bin_width
    item_length item_width item_id
    item_length item_width item_id
    ...
    item_length item_width item_id

    Make sure there's new line between instances
    '''

# Store each instance as an element in list
def read_dataset(filepath):
    instance_list = []
    
    # Read in file
    with open(filepath, "r") as rf:
        i = Instance()
        instance_list.append(i)
        for line in rf:
            spl = line.split()            
            if len(spl) == 0: # Blank line
                i = Instance()
                instance_list.append(i)
            elif len(spl) == 1: # Instance number
                continue
            elif len(spl) == 2: # Bin dimensions
                spl_int = [int(x) for x in spl] 
                i.set_binsize(spl_int)
            else: # Item dimensions and ID
                spl_int = [int(x) for x in spl]
                i.add_item(tuple(spl_int))
    rf.close()
    return instance_list

# Generate features from dataset
def generate_features(dataset):
    
    feature_space = []
    # Loops over each instance
    for i in dataset:

        # Generate features
        features = []
        
        # ITEM FEATURES
        areas = np.array([item[0]*item[1] for item in i.items])
        short_sides = np.array([min(item) for item in i.items])
        long_sides = np.array([max(item) for item in i.items])
        ratios = np.array([min(item)/max(item) for item in i.items])
        perimeters = np.array([(2*(item[0]+item[1])) for item in i.items])
        sides = np.append(short_sides, long_sides)

        # Number of items
        features.append(len(i.items))

        # Sum of item areas
        features.append(sum(areas))

        # Min item area
        features.append(min(areas))

        # Max item area
        features.append(max(areas))

        # Average item area
        features.append(np.mean(areas))

        # Standard deviation of item areas
        features.append(np.std(areas))

        # Variance of item areas
        features.append(np.var(areas))

        # Sum of item perimeters
        features.append(sum(perimeters))

        # Min item perimeter
        features.append(min(perimeters))

        # Max item perimeter
        features.append(max(perimeters))

        # Average item perimeter
        features.append(np.mean(perimeters))

        # Standard deviation of item perimeters
        features.append(np.std(perimeters))

        # Variance of item perimeters
        features.append(np.var(perimeters))

        # Sum of short sides
        features.append(sum(short_sides))

        # Min short side
        features.append(min(short_sides))

        # Max short side
        features.append(max(short_sides))

        # Average short side
        features.append(np.mean(short_sides))

        # Standard deviation of short sides
        features.append(np.std(short_sides))

        # Variance of short sides
        features.append(np.var(short_sides))

        # Sum of long sides
        features.append(sum(long_sides))

        # Min long side
        features.append(min(long_sides))

        # Max long side
        features.append(max(long_sides))

        # Average long side
        features.append(np.mean(long_sides))

        # Standard deviation of long sides
        features.append(np.std(long_sides))

        # Variance of long sides
        features.append(np.var(long_sides))

        # Sum of sides
        features.append(sum(sides))

        # Average side
        features.append(np.mean(sides))

        # Standard deviation of sides
        features.append(np.std(sides))

        # Variance of sides
        features.append(np.var(sides))

        # Sum of ratio of sides
        features.append(sum(ratios))

        # Min ratio of sides
        features.append(min(ratios))

        # Max ratio of sides
        features.append(max(ratios))

        # Average ratio of sides
        features.append(np.mean(ratios))

        # Standard deviation of ratios
        features.append(np.std(ratios))

        # Variance of ratios
        features.append(np.var(ratios))
        
        # BIN FEATURES
        # Area
        features.append(i.binsize[0] * i.binsize[1])

        # Short side
        features.append(min(i.binsize))

        # Long side
        features.append(max(i.binsize))

        # Ratio of sides
        features.append(min(i.binsize)/max(i.binsize))

        # CROSS FEATURES
        # Average item:bin area ratio
        features.append(np.mean(areas/(i.binsize[0] * i.binsize[1])))

        # Average item:bin short side
        features.append(np.mean(short_sides/min(i.binsize)))

        # Average item:bin long side
        features.append(np.mean(long_sides/max(i.binsize)))

        '''
        Idk add others maybe
        print(features)
        '''

        # Add to final feature labels
        feature_space.append(features) 

    return feature_space

