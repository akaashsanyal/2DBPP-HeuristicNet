import sys
import os
import random
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
def generate_raw_dataset(file_name, num_instances = 2000, max_boxes = 100, max_bin_length = 10, max_bin_width = 10):
    f = open(file_name, "a+")
    for inc in range(num_instances):
        f.write("+\n")
        num_boxes = random.randint(2, max_boxes)
        bin_length = random.randint(1, max_bin_length)
        bin_width = random.randint(1, max_bin_width)
        f.write(str(bin_length) + " " + str(bin_width) + "\n")
        for box in range(num_boxes):
            f.write(str(random.randint(1, bin_length)) + " " + str(random.randint(1, bin_width)) + " " + str(box) + "\n")
    f.close()

# Store each instance as an element in list
def read_dataset(filepath):
    instance_list = []
    
    # Read in file
    with open(filepath, "r") as rf:
        for line in rf:
            spl = line.split()            
            if len(spl) == 1: # Instance number
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
        box_perim = 2*(i.binsize[0]+i.binsize[1])
        box_area = i.binsize[0] * i.binsize[1]
        # Area
        features.append(box_area)
        
        # Perimeter
        features.append(box_perim)

        # Short side
        features.append(min(i.binsize))

        # Long side
        features.append(max(i.binsize))

        # Ratio of sides
        features.append(min(i.binsize)/max(i.binsize))

        # CROSS FEATURES
        # Min item:bin area ratio
        features.append(min(areas/box_area))
        
        # Max item:bin area ratio
        features.append(max(areas/box_area))
        
        # Average item:bin area ratio
        features.append(np.mean(areas/box_area))
        
        # Standard deviation item:bin area ratio
        features.append(np.std(areas/box_area))

        # Variance item:bin area ratio
        features.append(np.var(areas/box_area))
        
        # Min item:bin perimeter ratio
        features.append(min(perimeters/box_perim))
        
        # Max item:bin perimeter ratio
        features.append(max(perimeters/box_perim))
        
        # Average item:bin perimeter ratio
        features.append(np.mean(perimeters/box_perim))
        
        # Standard deviation item:bin perimeter ratio
        features.append(np.std(perimeters/box_perim))

        # Variance item:bin perimeter ratio
        features.append(np.var(perimeters/box_perim))

        # Min item:bin short side
        features.append(min(short_sides/min(i.binsize)))
        
        # Max item:bin short side
        features.append(max(short_sides/min(i.binsize)))
        
        # Average item:bin short side
        features.append(np.mean(short_sides/min(i.binsize)))
        
        # Standard deviation item:bin short side
        features.append(np.std(short_sides/min(i.binsize)))

        # Variance item:bin short side
        features.append(np.var(short_sides/min(i.binsize)))
        
        # Min item:bin long side
        features.append(min(long_sides/max(i.binsize)))
        
        # Max item:bin long side
        features.append(max(long_sides/max(i.binsize)))
        
        # Average item:bin long side
        features.append(np.mean(long_sides/max(i.binsize)))
        
        # Standard deviation item:bin long side
        features.append(np.std(long_sides/max(i.binsize)))

        # Variance item:bin long side
        features.append(np.var(long_sides/max(i.binsize)))

        # Add to final feature labels
        feature_space.append(features) 

    return feature_space

