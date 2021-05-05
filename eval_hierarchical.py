'''
Evaluation Approach (Always using model ensembles)
Use A1-C2 models to get grade predictions
Choose a threshold k
For all datapoints with predictions > k
Get new prediction using B2-C2 models
Caluclate MSE for all predictions
Repeat for all k to get a plot of MSE vs threshold k
'''
import sys
import os
import argparse
import torch
import torch.nn as nn
from tools import calculate_mse
import numpy as np
import matplotlib.pyplot as plt

def apply_hierarchal(preds_stage1, preds_stage2, thresh=4.0):
    preds = []
    for predA, predB in zip(preds_stage1, preds_stage2):
        if predA < thresh:
            preds.append(predA.item())
        else:
            preds.append(predB.item())
    return preds

def apply_hierarchal_ref(preds_stage1, preds_stage2, labels, thresh=4.0):
    '''
    Filter by true label
    '''
    preds = []
    for predA, predB, lab in zip(preds_stage1, preds_stage2, labels):
        if lab < thresh:
            preds.append(predA.item())
        else:
            preds.append(predB.item())
    return preds

def get_data(filename):
    preds = {}
    refs = {}

    with open(filename) as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = lines[1:] # exclude header

    for line in lines:
        items = line.split()
        id = items[0]
        ref = float(items[1])
        pred = float(items[2])

        preds[id] = pred
        refs[id] = ref

    return preds, refs


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILEA', type=str, help='First stage preds.txt file')
    commandLineParser.add_argument('FILEB', type=str, help='Second stage preds.txt file')
    commandLineParser.add_argument('OUT', type=str, help='Output .png file')

    args = commandLineParser.parse_args()
    fileA = args.FILEA
    fileB = args.FILEB
    out = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_hierarchical.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Store all data in dictionaries
    preds_stageB, _ = get_data(fileB)
    preds_stageA, refs_dict = get_data(fileA)

    # Align into lists
    predsA = []
    predsB = []
    refs = []

    for id in refs_dict:
        predsA.append(preds_stageA[id])
        predsB.append(preds_stageB[id])
        refs.append(refs_dict[id])

    # Make hierarchical plot

    ks = []
    rmses_hier = []
    rmses_ref = []
    rmses_baseline = []
    baseline = calculate_mse(torch.FloatTensor(predsA), torch.FloatTensor(refs)).item()
    baseline = baseline ** 0.5

    for k in np.linspace(0, 6, 60):
        preds = apply_hierarchal(torch.FloatTensor(predsA), torch.FloatTensor(predsB), thresh=k)
        mse_hier = calculate_mse(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
        rmse_hier = mse_hier**0.5
        preds_ref = apply_hierarchal_ref(torch.FloatTensor(predsA), torch.FloatTensor(predsB), torch.FloatTensor(refs), thresh=k)
        mse_ref = calculate_mse(torch.FloatTensor(preds_ref), torch.FloatTensor(refs)).item()
        rmse_ref = mse_ref**0.5
        ks.append(k)
        rmses_hier.append(rmse_hier)
        rmses_baseline.append(baseline)
        rmses_ref.append(rmse_ref)

    # Plot
    plt.plot(ks, rmses_baseline, label="Baseline")
    plt.plot(ks, rmses_hier, label="Hierarchical")
    plt.plot(ks, rmses_ref, label="Reference")
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(out)
    plt.clf()
