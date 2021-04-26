import torch
import sys
import os
import argparse
from tools import calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PREDS', type=str, help='pred.txt files separated by space')
    commandLineParser.add_argument('OUT', type=str, help='predicted scores file')

    args = commandLineParser.parse_args()
    pred_files = args.PREDS
    pred_files = pred_files.split()
    out_file = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_all.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Create spk_to_pred and spk_to_ref dicts
    pred_dicts = []
    ref_dicts = []
    for pred_file in pred_files:
        pred_dict = {}
        ref_dict = {}
        with open(pred_file) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]
        lines = lines[1:] # exclude header

        for line in lines:
            items = line.split()
            speakerid = items[0]
            ref = float(items[1])
            pred = float(items[2])
            pred_dict[speakerid] = pred
            ref_dict[speakerid] = ref

        pred_dicts.append(pred_dict)
        ref_dicts.append(ref_dict)

    # Form id, ref, pred lists
    speakerids = []
    refs = []
    preds = []

    for id in pred_dicts[0]:
        speakerids.append(id)
        ref_sum = 0
        ref_counter = 0


        for ref_dict in ref_dicts:
            try:
                ref = ref_dict[id]
                ref_sum += ref
                ref_counter += 1
            except:
                continue
        ref_overall = ref_sum/ref_counter
        refs.append(ref_overall)

        pred_sum = 0
        pred_counter = 0

        for pred_dict in pred_dicts:
            try:
                pred = pred_dict[id]
                pred_sum += pred
                pred_counter += 1
            except:
                continue
        pred_overall = pred_sum/pred_counter
        preds.append(pred_overall)

    # Get all the stats
    mse = calculate_mse(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
    pcc = calculate_pcc(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
    avg = calculate_avg(torch.FloatTensor(preds)).item()
    less05 = calculate_less05(torch.FloatTensor(preds), torch.FloatTensor(refs))
    less1 = calculate_less1(torch.FloatTensor(preds), torch.FloatTensor(refs))

    print("ALL PARTS STATS\n")
    print("MSE: ", mse)
    print("PCC: ", pcc)
    print("AVG: ", avg)
    print("LESS05: ", less05)
    print("LESS1: ", less1)

    # Save the predicted scores
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(speakerids, refs, preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
