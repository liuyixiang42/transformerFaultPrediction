from src.parser import *

output_folder = 'processed'
data_folder = 'data'

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
		'transformer': [(0.98, 1), (0.98, 1)],
	}
lm = lm_d[args.dataset][1]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001,
		'transformer': 0.001,
	}
lr = lr_d[args.dataset]

preds = []
