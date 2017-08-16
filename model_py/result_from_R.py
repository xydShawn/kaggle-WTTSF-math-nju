# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='file record the result from R', type=str)
parser.add_argument('-p', '--process', help='post-process the result, default: original', default='original', type=str)
args = parser.parse_args()

filename = '../result/' + args.filename
process_method = args.process
result = pd.read_csv(filename, header=None)
result = result.values
print('shape of result ', result.shape)

generate_submission_1(result, method=process_method)
