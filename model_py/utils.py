# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm



def strc(x):
    s = str(x) if x >= 10 else '0' + str(x)
    return s

def generate_submission_2(result):
    ttime = datetime.now()
    time2str = strc(ttime.month) + strc(ttime.day) + strc(ttime.hour) + strc(ttime.minute)
    filename = '../result/submission_' + time2str + '.csv'
    days = 60
    page_num = 145063
    if isinstance(result, pd.DataFrame):
        temp = result.values
    elif isinstance(result, np.ndarray):
        temp = result
    else:
        print('to do')
    submission_result = pd.read_csv('../data/sample_submission_1.csv')
    sub_res = submission_result.values
    map_pos = pd.read_csv('../tmp/map_pos.csv').values


    if temp.shape != (page_num, days):
        print('to do')
        return

    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Id', 'Visits'])
        for i in range(page_num * days):
            value = result[map_pos[i, 0], map_pos[i, 1]]
            writer.writerow([sub_res[i, 0], value])

def generate_submission_1(result):
    submission_result = pd.read_csv('../data/sample_submission_1.csv')
    map_pos = pd.read_csv('../tmp/map_pos.csv').values
    days = 60
    page_num = len(submission_result) // days

    if isinstance(result, pd.DataFrame):
        temp = result.values
    elif isinstance(result, np.ndarray):
        temp = result
    else:
        print('to do')

    if temp.shape == (page_num, days):
        visits = np.zeros((len(submission_result), 1))
        for i in tqdm(range(len(submission_result))):
            visits[i] = temp[map_pos[i, 0], map_pos[i, 1]]
        submission_result['Visits'] = visits
        ttime = datetime.now()
        time2str = strc(ttime.month) + strc(ttime.day) + strc(ttime.hour) + strc(ttime.minute)
        filename = '../result/submission_' + time2str + '.csv'
        submission_result.to_csv(filename, index=False)
    else:
        print('to do')

    
