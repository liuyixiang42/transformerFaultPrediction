from parser import *


output_folder = 'processed'
data_folder = 'data'



# Threshold parameters
lm_d = {
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMAP': 0.001,
		'MSL': 0.002,
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMAP': (97, 5000),
		'MSL': (97, 150),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
