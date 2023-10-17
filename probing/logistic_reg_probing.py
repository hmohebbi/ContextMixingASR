import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--ARCH")
parser.add_argument("--LAYER", type=int)
parser.add_argument("--RANDOM", action='store_true')
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
ARCH = args.ARCH
LAYER = args.LAYER
RANDOM = args.RANDOM

# MODEL_NAME = "whisper-medium" 
# ARCH = "enc"
# LAYER = 3
# RANDOM = False

TASK = "common_voice"
SPLIT = "test" 
TEMPLATE = "all"
SEED = 12
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"
REPRESENTATIONS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/representations/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"
SAVE_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/probes/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"

# Import Packages
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
from tqdm.auto import tqdm
import pickle
import json
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from datasets import load_from_disk, Dataset
from utils import MODEL_PATH

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

LABEL_MAPPER = {'Number_sing': 0, 'Number_plur': 1}
is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"

# load annotated data
annot_data = load_from_disk(f"{ANNOTATED_DATA_PATH}{TEMPLATE}")
annot_data = annot_data.add_column("ex", list(range(len(annot_data))))
unique_org_id = [i for i in annot_data['org_id'] if annot_data['org_id'].count(i) == 1]
unique_ids = [i for i, x in enumerate(annot_data['org_id']) if x in unique_org_id]
annot_data = annot_data.select(unique_ids)
# 
D = annot_data.select(np.where(np.array(annot_data['template']) == "det_noun_verb")[0])
for x in D:
    temp = x
    temp['cue_indices']['enc'] = x['cue_indices']['enc']  + x['target_indices']['enc'] 
    temp['target_indices']['enc']  = x['target_indices_2']['enc'] 
    if is_encoder_decoder:
        temp['cue_indices']['dec'][MODEL_NAME] = x['cue_indices']['dec'][MODEL_NAME]  + x['target_indices']['dec'][MODEL_NAME]
        temp['target_indices']['dec'][MODEL_NAME]  = x['target_indices_2']['dec'][MODEL_NAME]
    annot_data = annot_data.add_item(temp)
#
num_labels = 2
all_labels = annot_data['label_number'] 

postfix = ""
if RANDOM:
    postfix += "_random"
with open(f'{REPRESENTATIONS_PATH}{ARCH}_target_representations{postfix}.pkl', 'rb') as fp:
    all_reps = pickle.load(fp)
data = {'representations': [], 'labels': []}
for i in range(len(all_labels)):
    data['representations'].append(all_reps[i][LAYER-1])
    data['labels'].append(LABEL_MAPPER[all_labels[i]])
data = Dataset.from_dict(data).shuffle(SEED)
num_examples = len(data)

# run
clf = LogisticRegression(solver="liblinear", penalty='l2', random_state=42)
scores = cross_validate(clf, np.array(data['representations']), data['labels'], cv=3, scoring=('accuracy', 'r2', 'neg_mean_squared_error', 'f1_macro'), return_train_score=True)
test_acc = scores['test_accuracy'].mean().item()
test_f1 = scores['test_f1_macro'].mean().item()
print("Layer:", LAYER)
print("train accuracy: ", scores['train_accuracy'])
print("test accuracy: ", scores['test_accuracy'])
print("train f1_macro: ", scores['train_f1_macro'])
print("test f1_macro: ", scores['test_f1_macro'])

# Save
with open(f'{SAVE_PATH}acc_lr_{ARCH}_{LAYER}{postfix}.json', 'w') as f:
    json.dump(test_acc, f)
with open(f'{SAVE_PATH}f1_lr_{ARCH}_{LAYER}{postfix}.json', 'w') as f:
    json.dump(test_f1, f)



