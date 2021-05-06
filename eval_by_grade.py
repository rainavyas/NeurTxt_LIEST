'''
Calculates statistics (e.g. MSE, PCC) using a id, ref, pred .txt
file (e.g. output by eval_all_calibrate.py).
Calculates statistics by grade, i.e. A1, A2, B1, B2, C1, C2
A1: 0.5-1.5
A2: 1.5-2.5
B1: 2.5-3.5
B2: 3.5-4.5
C1: 4.5-5.5
C2: 5.5-6.0
'''

import torch
import sys
import os
import argparse
from tools import calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg

class DataHandler():
    '''
    For dealing with any statistic caluclations for a dataset
    '''
    def __init__(self, ids=None, preds=None, refs=None):
        if ids is None:
            self.ids = []
            self.preds = []
            self.refs = []
        else:
            self.ids = ids
            self.preds = preds
            self.refs = refs

    def add_data(self, id, pred, ref):
        self.ids.append(id)
        self.preds.append(pred)
        self.refs.append(ref)

    def all_stats(self):
        mse = calculate_mse(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs)).item()
        pcc = calculate_pcc(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs)).item()
        avg = calculate_avg(torch.FloatTensor(self.preds)).item()
        less05 = calculate_less05(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs))
        less1 = calculate_less1(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs))

        return mse, pcc, avg, less05, less1


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PRED', type=str, help='overall pred.txt file: id ref pred')

    args = commandLineParser.parse_args()
    pred_file = args.PRED

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_by_grade.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data
    with open(pred_file) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = lines[1:] # exclude header

    grade_boundaries = {"A1":0.5, "A2":1.5, "B1":2.5, "B2":3.5, "C1":4.5, "C2":5.5}

    for grade, val in grade_boundaries:
        handler = DataHandler()
        for line in lines:
            items = line.split()
            ref = float(items[1])
            if ref >= val and ref < val+1:
                handler.add_data(items[0], float(items[2]), ref)
        mse, pcc, avg, less05, less1 = handler.all_stats()

        print("Grade", grade)
        print("MSE:", mse)
        print("PCC:", pcc)
        print("AVG:", avg)
        print("Less 0.5:", less05)
        print("Less 1.0", less1)
        print()

    # Get overall statistics
    handler = DataHandler()
    for line in lines:
        items = line.split()
        ref = float(items[1])
        handler.add_data(items[0], float(items[2]), ref)
    mse, pcc, avg, less05, less1 = handler.all_stats()

    print("Overall Statistics")
    print("MSE:", mse)
    print("PCC:", pcc)
    print("AVG:", avg)
    print("Less 0.5:", less05)
    print("Less 1.0", less1)
    print()
