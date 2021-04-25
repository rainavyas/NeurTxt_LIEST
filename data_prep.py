'''
Prepares the data in tensor format
'''
import torch
import torch.nn as nn
from transformers import BertTokenizer

def get_spk_to_utt(responses_file, part):

    # Load the responses
    with open(responses_file, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    # Concatenate utterances for a speaker
    spk_to_utt = {}
    for line in lines:
        speaker_part = int(line[14])
        if speaker_part != part:
            continue
        speakerid = line[:12]
        utt = line[20:]

        if speakerid not in spk_to_utt:
            spk_to_utt[speakerid] = utt
        else:
            spk_to_utt[speakerid] = spk_to_utt[speakerid] + ' ' + utt
    return spk_to_utt

def get_spk_to_grade(grades_file, part):

    # Load the grades
    with open(grades_file, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    grade_dict = {}
    for line in lines:
        speaker_part = int(line[14])
        if speaker_part != part:
            continue
        speakerid = line[:12]
        grade = float(line[-3:])
        grade_dict[speakerid] = grade
    return grade_dict

def align(spk_to_utt, grade_dict):
    grades = []
    utts = []
    speakerids = []
    for id in spk_to_utt:
        try:
            grades.append(grade_dict[id])
            utts.append(spk_to_utt[id])
            speakerids.append(id)
        except:
            print("Falied for speaker " + str(id))
    return utts, grades, speakerids

def tokenize_text(utts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(utts, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return ids, mask


def get_data(responses_file, grades_file, part=1):

    spk_to_utt = get_spk_to_utt(responses_file, part)
    grade_dict = get_spk_to_grade(grades_file, part)
    utts, grades, speaker_ids = align(spk_to_utt, grade_dict)
    input_ids, mask = tokenize_text(utts)
    labels = torch.FloatTensor(grades)
    return input_ids, mask, labels, speaker_ids
