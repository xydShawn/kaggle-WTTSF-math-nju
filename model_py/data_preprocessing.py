# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def start_idx(w):
    nan_idx = np.where(np.isnan(w))[0]
    cnt = 0
    while cnt < len(nan_idx) and nan_idx[cnt] == cnt:
        cnt += 1
    return cnt

def start_inter(matrix, idx_interpt, interpolate_kind, page_num, days):
	new_matrix = np.copy(matrix)
	start_dict = np.zeros((page_num,), dtype=np.int)
	x = np.arange(0, days)
	for i in range(len(idx_interpt)):
		i_real = idx_interpt[i]
		w = matrix[i_real, :]
		start = start_idx(w)
		start_dict[i_real] = start 
		w[0:start] = 0
		yy_idx = np.where(np.isnan(w))[0]
		yy = np.delete(w, yy_idx)
		xx = np.delete(x, yy_idx)
		if interpolate_kind in ['linear', 'nearest']:
			f = interp1d(xx, yy, kind = interpolate_kind, fill_value = "extrapolate")
		elif interpolate_kind in ['zero', 'slinear', 'quadratic', 'cubic']:
			if (xx[-1] != days - 1):
				xx = np.append(xx, days - 1)
				yy = np.append(yy, 0)
			f = interp1d(xx, yy, kind = interpolate_kind)
		else:
			print 'kind does not exist!'
			return 
		ynew = f(x)
		new_matrix[i_real, :] = ynew
	print("Interpolate is done.")
	return new_matrix, start_dict

def get_NaN_idx(matrix, days):
	z = np.where(np.isnan(matrix))
	delete_nan_row = {}
	for x in z[0]:
		if x in delete_nan_row:
			delete_nan_row[x] += 1
		else:
			delete_nan_row[x] = 1
	idx_interpt = [k for k, v in delete_nan_row.iteritems() if v < days]
	idx_all_nan = [k for k, v in delete_nan_row.iteritems() if v == days]
	print "# of lines needed to be interpolate: ", len(idx_interpt)
	print "# of lines which are all NaN: ", len(idx_all_nan)
	return idx_interpt, idx_all_nan


def clean_NaN(treated, start_date, end_date, interpolate_kind, genDict, remove_peaks, percentile, times):
	print("start loading data...")
	if treated == False:
		df1 = pd.read_csv("../data/train_1.csv")
		matrix = df1.iloc[:, start_date:end_date].values
		print "matrix shape: ", matrix.shape
		(page_num, days) = matrix.shape
		idx_interpt, idx_all_nan = get_NaN_idx(matrix, days)
		if remove_peaks == True:
			idx = np.arange(page_num)
			idx = np.delete(idx, idx_all_nan)
			peaks = np.nanpercentile(matrix[idx, :], percentile, axis=1)
			peaks = times * peaks
			for i in range(len(idx)):
				ii = idx[i]
				matrix[ii][matrix[ii, :] > peaks[i]] = np.nan
				matrix = np.around(matrix)
				np.savetxt("../result/remove_peaks.csv.gz", matrix, fmt='%s', delimiter=",")
				print 'removing peaks is done.', matrix.shape
	else:
		matrix = np.genfromtxt("../result/remove_peaks.csv.gz", delimiter=",")
		(page_num, days) = matrix.shape
		idx_interpt, idx_all_nan = get_NaN_idx(matrix, days)

	new_matrix, start_dict = start_inter(matrix, idx_interpt, interpolate_kind, page_num, days)
	new_matrix[idx_all_nan, :] = 0
	if genDict == True:
		start_idx_Name = "../result/first_non_NaN_start_idx" + str(end_date - 1) + ".csv"
		np.savetxt(start_idx_Name, start_dict, fmt='%i', delimiter=",")
	interp_name = "../result/clean_NaN_" + interpolate_kind + ".csv.gz"
	print 'start saving file...'
	np.savetxt(interp_name, new_matrix, fmt='%i', delimiter=",")


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--treated', help='treated, default=True', default=True, type=bool)
parser.add_argument('-s', '--start_date', help='start_date, default=1', default=1, type=int)
parser.add_argument('-e', '--end_date', help='end_date, default = 491', default=491, type=int)
parser.add_argument('-i', '--interpolate_kind', help='interpolate_kind, choose from linear, nearest, zero, slinear, quadratic, cubic. default: linear', default='linear', type=str)
parser.add_argument('-g', '--genDict', help='decide if write start index', default=False, type=bool)
parser.add_argument('-r', '--remove_peaks', help='remove_peaks, default=True', default=True, type=bool)
parser.add_argument('-p', '--percentile', help='percentile, default = 75', default=75, type=int)
parser.add_argument('-t', '--times', help='times, default = 10', default=10, type=int)
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date
interpolate_kind = args.interpolate_kind
genDict = args.genDict
remove_peaks = args.remove_peaks
percentile = args.percentile
times = args.times
treated = args.treated

clean_NaN(treated, start_date, end_date, interpolate_kind, genDict, remove_peaks, percentile, times)
