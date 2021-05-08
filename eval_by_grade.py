'''
Calculates statistics (e.g. MSE, PCC) using a id, ref, pred .txt
file (e.g. output by eval_all_calibrate.py).
Calculates statistics by grade, i.e. A1, A2, B1, B2, C1, C2
Pre A1: 0 <= y < 1
A1: 1 <= y < 2
A2: 2 <= y < 3
B1: 3 <= y < 4
B2: 4 <= y < 5
C1-C2: 5 <= y < 6
'''

import torch
import sys
import os
import argparse
from tools import calculate_rmse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg

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
        rmse = calculate_rmse(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs)).item()
        pcc = calculate_pcc(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs)).item()
        avg = calculate_avg(torch.FloatTensor(self.preds)).item()
        less05 = calculate_less05(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs))
        less1 = calculate_less1(torch.FloatTensor(self.preds), torch.FloatTensor(self.refs))

        return rmse, pcc, avg, less05, less1


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

    grade_boundaries = {"Pre A1":0.0, "A1":1.0, "A2":2.0, "B1":3.0, "B2":4.0, "C1-C2":5.0}

    for grade, val in grade_boundaries.items():
        handler = DataHandler()
        for line in lines:
            items = line.split()
            ref = float(items[1])
            if ref >= val and ref < val+1:
                handler.add_data(items[0], float(items[2]), ref)
        num_spk = len(handler.ids)
        print("Grade", grade)
        print("Number of speakers:", num_spk)
        if num_spk == 0:
            print()
            continue
        rmse, pcc, avg, less05, less1 = handler.all_stats()

        print("RMSE:", rmse)
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
    rmse, pcc, avg, less05, less1 = handler.all_stats()
    num_spk = len(handler.ids)

    print("Overall Statistics")
    print("Number of speakers:", num_spk)
    print("RMSE:", rmse)
    print("PCC:", pcc)
    print("AVG:", avg)
    print("Less 0.5:", less05)
    print("Less 1.0", less1)
    print()
