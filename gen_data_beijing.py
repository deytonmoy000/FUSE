
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import collections
from datetime import datetime, timedelta
import time
import random
import json
import h5py

import tensorflow as tf

import concurrent.futures
import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from queue import PriorityQueue
from utils_beijing import gen_granular_data

if __name__ == "__main__":

    grid_size = int(sys.argv[1])

    grid_size_fname = str(grid_size)

    print("\n********-----********-----********-----********-----*******\n")
    print("Generating Data File for Grid Size: {} ...".format(grid_size))

    gen_granular_data(grid_size)

    print("Data Generation Complete")

