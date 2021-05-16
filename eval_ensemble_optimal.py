'''
Takes in Model A and Model B predictions

predicted grade = a*ModelA_pred + (1-a)*ModelB_pred
a is optimised with respect to rmse to reference grades
'''

import sys
import os
import argparse
import torch
from eval_hierarchical import get_data
from tools import calculate_rmse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('FILEA', type=str, help='Model A preds.txt file')
    commandLineParser.add_argument('FILEB', type=str, help='Model B preds.txt file')
    commandLineParser.add_argument('OUT', type=str, help='Output predictions file')

    args = commandLineParser.parse_args()
    fileA = args.FILEA
    fileB = args.FILEB
    out_file = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_ensemble_optimal.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Store all data in dictionaries
    preds_modelB, _ = get_data(fileB)
    preds_modelA, refs_dict = get_data(fileA)

    # Align into lists
    predsA = []
    predsB = []
    refs = []
    ids = []

    for id in refs_dict:
        try:
            predsB.append(preds_modelB[id])
            predsA.append(preds_modelA[id])
            refs.append(refs_dict[id])
            ids.append(id)
        except:
            print("ID mismatch ", id)

    # Find optimal linear combination
    best_alpha = 0.5
    best_rmse = 100

    alphas = []
    rmses = []

    for alpha in np.linspace(0,1,50):
        new_preds = [(alpha*predA + (1-alpha)*predB) for predA, predB in zip(predsA, predsB)]
        rmse = calculate_rmse(torch.FloatTensor(new_preds), torch.FloatTensor(refs)).item()

        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse

        alphas.append(alpha)
        rmses.append(rmse)

    # Plot results
    filename = 'ensemble_optimal.png'
    plt.plot(alphas, rmses)
    plt.xlabel('Ensemble split ratio')
    plt.ylabel('RMSE')
    plt.savefig(filename)
    plt.clf()

    # Save the optimal predicted scores
    alpha = best_alpha
    preds = [(alpha*predA + (1-alpha)*predB) for predA, predB in zip(predsA, predsB)]
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(ids, refs, preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
