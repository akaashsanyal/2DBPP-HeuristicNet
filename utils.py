import sys
import os

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
def generate_dataset(filepath):
    '''
    Format is as follows:
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
    with open(filepath, "r") as rf:
        i = Instance()
        instance_list.append(i)
        for line in rf:
            spl = line.split()            
            if len(spl) < 2:
                i = Instance()
                instance_list.append(i)
            elif len(spl) == 2:
                spl_int = [int(x) for x in spl] 
                i.set_binsize(tuple(spl_int))
            else:
                spl_int = [int(x) for x in spl]
                i.add_item(tuple(spl_int))
    rf.close()
    return instance_list

