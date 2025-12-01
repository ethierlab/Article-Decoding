import pickle

"""
Please note:
For each monkey Pop's data file in this folder, M1 neural firing rates have been
binned (50 ms bins) and smoothed (Gaussian kernel SD = 100 ms). Please DO NOT
bin and smooth the `spike_counts` again.

For each monkey Pop's data file in this folder, the outliers and baseline offsets 
in all EMG channels have been removed, and all EMG samples have been normalized.
Please DO NOT do these preprocessing again.

For each monkey Pop's data file in this folder, all trials have been splitted, but
not time aligned. Each trial started at the 'gocue' time, and ended at the 'end' time
(namely reward time). 

In summary, for monkey Pop please use the data loaded from the .pkl files directly
when doing your own analysis. NO NEED for any pre-processing.
"""

file_name = 'Pop_20210602_pg.pkl'

EMG_names = ['APB', 'FPB', 'Lum', 'PT', 'FDS1', 'FDP2', '1DI', '4DI', 'EPM', 'ECR', 'EDC1']

with open(file_name, 'rb') as fp:
    spike_counts = pickle.load(fp)
    EMG = pickle.load(fp)
    force = pickle.load(fp)