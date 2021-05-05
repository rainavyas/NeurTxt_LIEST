'''
Same as calibrate.py but calibration coefficients only
calculated using data points filtered by reference grade
'''
from statistics import mean
import numpy as np
import sys
import os
import argparse
from calibrate import best_fit_slope_and_intercept

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PRED', type=str, help='overall pred.txt file: id ref pred')
    commandLineParser.add_argument('OUT', type=str, help='to store calibration results')
    commandLineParser.add_argument('--filter', type=float, default=4.0, help="Specify grade threshold")

    args = commandLineParser.parse_args()
    pred_file = args.PRED
    out_file = args.OUT
    filter = args.filter

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/calibrate_filter.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data
    with open(pred_file) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = lines[1:] # exclude header

    ids = []
    preds = []
    refs = []

    for line in lines:
        items = line.split()
        # skip if reference grade too low
        if float(items[1]) < filter:
            continue
        ids.append(items[0])
        refs.append(float(items[1]))
        preds.append(float(items[2]))

    preds = np.array(preds)
    refs = np.array(refs)
    print(len(preds))

    m, b = best_fit_slope_and_intercept(preds,refs)
    print("gradient:", m)
    print("y-intercept:", b)

    # write results of calibration
    with open(out_file, 'w') as f:
        text = "Filtered Calibration results using " + pred_file + "\n gradient: " +str(m) + "\n y-intercept: "+str(b)
        f.write(text)
