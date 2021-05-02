'''
Calibration by a linear transformation
Returns calibration coefficients (mean and y-intercept)
'''
from statistics import mean
import numpy as np
import sys
import os
import argparse

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)

    return m, b

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PRED', type=str, help='overall pred.txt file: id ref pred')
    commandLineParser.add_argument('OUT', type=str, help='to store calibration results')

    args = commandLineParser.parse_args()
    pred_file = args.PRED
    out_file = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/calibrate.cmd', 'a') as f:
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
        ids.append(items[0])
        refs.append(float(items[1]))
        preds.append(float(items[2]))

    preds = np.array(preds)
    refs = np.array(refs)

    m, b = best_fit_slope_and_intercept(preds,refs)
    print("gradient:", m)
    print("y-intercept:", b)

    # write results of calibration
    with open(out_file, 'w') as f:
        text = "Calibration results using " + pred_file + "\n gradient: " +str(m) + "\n y-intercept: "+str(b)
        f.write(text)
