'''
Ensemble predictions of model A and model B

predicted grade = a*ModelA_pred + (1-a)*ModelB_pred
where,
a = linear_function(ref_grade)

The linear function is optimised  (two parameters to optimise)
to minimise rmse after ensembling
'''

import sys
import os
import argparse
import torch
from eval_hierarchical import get_data
from tools import calculate_rmse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
    with open('CMDs/eval_ensemble_linear_alpha.cmd', 'a') as f:
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

    # Find optimal linear mapper from ref to alpha
    best_m = 1.0 # gradient of mapper
    best_c = 0.0 # shift of mapper
    best_rmse = 100

    ms = np.linspace(-1,1,100)
    cs = np.linspace(-1,1,100)
    rmses = np.zeros((len(ms), len(cs)))

    for i,m in enumerate(ms):
        for j,c in enumerate(cs):
            new_preds = [((m*ref+c)*predA + (1-(m*ref+c))*predB) for predA, predB, ref in zip(predsA, predsB, refs)]
            rmse = calculate_rmse(torch.FloatTensor(new_preds), torch.FloatTensor(refs)).item()
            if rmse < best_rmse:
                best_m = m
                best_c = c
                best_rmse = rmse
            rmses[i,j] = rmse

    # Plot the results
    filename = 'ensemble_linear_alpha.png'
    ms, cs = np.meshgrid(ms, cs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(ms,cs,rmses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0.395, 0.44)
    fig.colorbar(surf)
    plt.savefig(filename)
    plt.clf()

    # Save the optimal predicted scores
    m = best_m
    c = best_c
    print("m", m)
    print("c", c)
    preds = [((m*ref+c)*predA + (1-(m*ref+c))*predB) for predA, predB, ref in zip(predsA, predsB, refs)]
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(ids, refs, preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
