import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep import get_data
import sys
import os
import argparse
from tools import AverageMeter, get_default_device, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg
from models import BERTGrader
import statistics

def eval(val_loader, model, device):
    targets = []
    preds = []

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (id, mask, target) in enumerate(val_loader):

            id = id.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            pred = model(id, mask)

            # Store
            preds += pred.tolist()
            targets += target.tolist()
    return preds, targets

def get_single_stats(all_preds, targets):
    mses = []
    pccs = []
    avgs = []
    less05s = []
    less1s = []

    for preds in all_preds:
        mses.append(calculate_mse(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())
        pccs.append(calculate_pcc(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())
        avgs.append(calculate_avg(torch.FloatTensor(preds)).item())
        less05s.append(calculate_less05(torch.FloatTensor(preds), torch.FloatTensor(targets)))
        less1s.append(calculate_less1(torch.FloatTensor(preds), torch.FloatTensor(targets)))

    mse_mean = statistics.mean(mses)
    mse_std = statistics.pstdev(mses)
    pcc_mean = statistics.mean(pccs)
    pcc_std = statistics.pstdev(pccs)
    avg_mean = statistics.mean(avgs)
    avg_std = statistics.pstdev(avgs)
    less05_mean = statistics.mean(less05s)
    less05_std = statistics.pstdev(less05s)
    less1_mean = statistics.mean(less1s)
    less1_std = statistics.pstdev(less1s)

    return mse_mean, mse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std

def get_ensemble_stats(all_preds, targets):
    y_sum = torch.zeros(len(all_preds[0]))
    for preds in all_preds:
        y_sum += torch.FloatTensor(preds)
    ensemble_preds = y_sum/len(all_preds)

    mse = calculate_mse(ensemble_preds, torch.FloatTensor(targets))
    pcc = calculate_pcc(ensemble_preds, torch.FloatTensor(targets))
    avg = calculate_avg(ensemble_preds)
    less05 = calculate_less05(ensemble_preds, torch.FloatTensor(targets))
    less1 = calculate_less1(ensemble_preds, torch.FloatTensor(targets))

    return ensemble_preds.tolist(), mse.item(), pcc.item(), avg.item(), less05, less1


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELS', type=str, help='trained .th models separated by space')
    commandLineParser.add_argument('RESPONSES', type=str, help='responses text file')
    commandLineParser.add_argument('GRADES', type=str, help='scores text file')
    commandLineParser.add_argument('OUT', type=str, help='predicted scores file')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")

    args = commandLineParser.parse_args()
    model_paths = args.MODELS
    model_paths = model_paths.split()
    responses_file = args.RESPONSES
    out_file = args.OUT
    grades_file = args.GRADES
    batch_size = args.B
    part=args.part

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_ensemble.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data
    input_ids, mask, labels, speakerids = get_data(responses_file, grades_file, part=part)
    test_ds = TensorDataset(input_ids, mask, labels)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Load the models
    models = []
    for model_path in model_paths:
        model = BERTGrader()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        models.append(model)

    targets = None
    all_preds = []

    for model in models:
        preds, targets = eval(test_dl, model, device)
        all_preds.append(preds)

    # Get single stats
    mse_mean, mse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std = get_single_stats(all_preds, targets)
    print("STATS FOR ", model_paths)
    print()
    print("SINGLE STATS\n")
    print("MSE: "+str(mse_mean)+" +- "+str(mse_std))
    print("PCC: "+str(pcc_mean)+" +- "+str(pcc_std))
    print("AVG: "+str(avg_mean)+" +- "+str(avg_std))
    print("LESS05: "+str(less05_mean)+" +- "+str(less05_std))
    print("LESS1: "+str(less1_mean)+" +- "+str(less1_std))

    # Get ensemble stats
    ensemble_preds, mse, pcc, avg, less05, less1 = get_ensemble_stats(all_preds, targets)
    print()
    print("ENSEMBLE STATS\n")
    print("MSE: ", mse)
    print("PCC: ", pcc)
    print("AVG: ", avg)
    print("LESS05: ", less05)
    print("LESS1: ", less1)

    # Save the predicted scores
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(speakerids, targets, ensemble_preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
