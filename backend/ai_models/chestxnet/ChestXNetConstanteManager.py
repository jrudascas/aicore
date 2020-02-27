import os
import sys


LABEL_BASELINE_PROBS = {
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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PATH_WEIGHTS = os.path.dirname(sys.modules['__main__'].__file__) + '/backend/ai_models/chestxnet/pretrained/checkpoint'