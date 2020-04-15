import sys
import os
import numpy as np
import pandas as pd

class Instance:
    def __init__(self, binsize=(0,0), items=[]):
        self.binsize = binsize
        self.items = items

    def add_item(self, item):
        self.items.append(item)
    

# Store each instance as an element in list
def read_dataset(filepath):
    # Decide format of data
    # I'm thinking bin size (width, length)
    # Then list of items with (id, width, length)
    instance_list = []

