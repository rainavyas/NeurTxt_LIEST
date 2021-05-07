'''
Creates a prediction file for DDN grader in the following format:
id ref pred

reference predictions from one of my existing files prediction files

Excluding section 2 predictions to be consistent
'''

import sys
import os
import argparse

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DDN_PRED', type=str, help='DDN prediction file')
    commandLineParser.add_argument('REF_PRED', type=str, help= 'id ref pred .txt file for reference')
    commandLineParser.add_argument('OUT', type=str, help='output predicted scores file')

    args = commandLineParser.parse_args()
    ddn_pred_file = args.DDN_PRED
    ref_pred_file = args.REF_PRED
    out_file = args.OUT

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/tranform_DDN.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Create speaker to ref dict
    ref_dict = {}
    with open(ref_pred_file, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = lines[1:] # exclude header

    for line in lines:
        items = line.split()
        ref_dict[items[0]] = float(items[1])

    # Create ids, refs and preds aligned lists
    with open(ddn_pred_file, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    ids = []
    refs = []
    preds = []

    for line in lines:
        items = line.split()
        id = items[0]

        parts = [1,3,4,5]
        total = 0
        count = 0

        for part in parts:
            try:
                total += float(items[part])
                count +=1
            except:
                continue
        pred = total/count

        try:
            ref = ref_dict[id]
        except:
            print("ID not found", id)

        ids.append(id)
        refs.append(ref)
        preds.append(pred)

    # Write to out file
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(ids, refs, preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
