# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils import *



result = pd.read_csv('../result/ts_for_all.csv', header=None)
result = result.values
print('shape of result ', result.shape)

generate_submission_1(result)
