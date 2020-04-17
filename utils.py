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
            if len(spl) < 2: # Blank line
                i = Instance()
                instance_list.append(i)
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
        
        # Item features
        vols = [item[0]*item[1] for item in i.items]
        print(vols)
        # Number of items
        features.append(len(i.items))

        # Sum of item volumes
        features.append(sum(vols))

        # Min item volume
        features.append(min(vols))

        # Max item volume
        features.append(max(vols))

        # Average item volume
        features.append(np.mean(vols))

        # Standard deviation of item volumes
        features.append(np.std(vols))

        # Variance of item volumes
        features.append(np.var(vols))
       
        '''
        Idk add others
        '''
        print(features)
        # Add to final feature labels
        feature_space.append(features) 

    return feature_space

