import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from data_prep import get_data
import sys
import os
import argparse
from tools import AverageMeter, get_default_device
from models import BERTGrader
from train import train, eval

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('RESPONSES', type=str, help='responses text file')
    commandLineParser.add_argument('GRADES', type=str, help='scores text file')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=3, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler param")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")
    commandLineParser.add_argument('--val_size', type=int, default=200, help="Specify validation set size")
    commandLineParser.add_argument('--filter', type=float, default=4.0, help="Specify grade threshold")



    args = commandLineParser.parse_args()
    out_file = args.OUT
    responses_file = args.RESPONSES
    grades_file = args.GRADES
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    sch = args.sch
    seed = args.seed
    part = args.part
    val_size = args.val_size
    filter = args.filter


    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_filter.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data
    input_ids, mask, labels, _ = get_data(responses_file, grades_file, part=part)
    print(mask.size())

    # filter data to only keep grades equal to or above the filter value
    filtered = labels >= filter
    inds = filtered.nonzero().squeeze()

    input_ids = input_ids[inds]
    mask = mask[inds]
    labels = labels[inds]
    print(mask.size())

    # split into training and validation sets
    input_ids_val = input_ids[:val_size]
    mask_val = mask[:val_size]
    labels_val = labels[:val_size]

    input_ids_train = input_ids[val_size:]
    mask_train = mask[val_size:]
    labels_train = labels[val_size:]

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids_train, mask_train, labels_train)
    val_ds = TensorDataset(input_ids_val, mask_val, labels_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # initialise grader
    model = BERTGrader()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[sch])

    # Criterion
    criterion = torch.nn.MSELoss(reduction='mean')

    # Train
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate as we go along
        eval(val_dl, model, criterion, device)

    # Save the trained model
    state = model.state_dict()
    torch.save(state, out_file)
