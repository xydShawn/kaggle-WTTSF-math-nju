# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime



def strc(x):
    s = str(x) if x >= 10 else '0' + str(x)
    return s

def generate_submission_1(result):
    submission_result = pd.read_csv('./data/sample_submission_1.csv')
    map_pos = pd.read_csv('./tmp/map_pos.csv').values
    days = 60
    page_num = len(submission_result) // days

    if isinstance(result, pd.DataFrame):
        temp = result.values
    elif isinstance(result, np.array):
        temp = result
    else:
        print('to do')

    if temp.shape == (page_num, days):
        for i in range(len(submission_result)):
            submission_result.loc[i, 'Visits'] = temp[map_pos[i, 0], map_pos[i, 1]]
        ttime = datetime.now()
        time2str = strc(ttime.month) + strc(ttime.day) + strc(ttime.hour) + strc(ttime.minute)
        filename = './result/submission_' + time2str + '.csv'
        submission_result.to_csv(filename, index=False)
    else:
        print('to do')

    
