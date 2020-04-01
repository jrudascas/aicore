import os
import sys
import collections


INITIAL_LABEL_BASELINE_PROBS = {
    'Atelectasis': 0.103,
    'Cardiomegaly': 0.025,
    'Effusion': 0.119,
    'Infiltration': 0.177,
    'Mass': 0.051,
    'Nodule': 0.056,
    'Pneumonia': 0.012,
    'Pneumothorax': 0.047,
    'Consolidation': 0.042,
    'Edema': 0.021,
    'Emphysema': 0.022,
    'Fibrosis': 0.015,
    'Pleural_Thickening': 0.03,
    'Hernia': 0.002
}

LABEL_BASELINE_PROBS = collections.OrderedDict(INITIAL_LABEL_BASELINE_PROBS)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PATH_SAVE_VISUAL_RESPONSE = '/tmp/'
PATH_WEIGHTS = os.path.abspath(os.curdir) + '/backend/ai_models/chestxnet/pretrained/checkpoint'