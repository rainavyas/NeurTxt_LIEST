# Description

Text based grader for spoken language assessment on LIEST data.

# Model Architecture

Transformer encoder followed by an extra multihead attention layer and a DNN for regression style grade prediction from ASR transcribed utterance responses by a speaker to set prompts.

# Requirements

Python3.6 or above

## Install with PyPI

pip install torch
 
pip install transformers
