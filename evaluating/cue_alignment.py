import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--DIM_AGGREGATOR")
parser.add_argument("--SKIP_SPECIAL_TOKENS", action='store_true')
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
DIM_AGGREGATOR = args.DIM_AGGREGATOR
SKIP_SPECIAL_TOKENS = args.SKIP_SPECIAL_TOKENS

# MODEL_NAME = "whisper-medium" 
# DIM_AGGREGATOR = "mean"
# SKIP_SPECIAL_TOKENS = False

TEMPLATE = "all"
TASK = "common_voice"
SPLIT = "test" 
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"
GENERATED_IDS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/{MODEL_NAME}/"
SCORES_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/scores/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"
SAVE_EVAL_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/evals/cue_alignment/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"

# imports
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import pickle
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import WhisperProcessor, Wav2Vec2Processor
from utils import MODEL_PATH, NUM_LAYERS

if not os.path.exists(SAVE_EVAL_PATH):
    os.makedirs(SAVE_EVAL_PATH)

is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"
METHOD_NAMES = [
                f'encoder_attention_{DIM_AGGREGATOR}_random',
                f'encoder_summedNorms_{DIM_AGGREGATOR}_random',
                f'encoder_valueZeroing-cosine_{DIM_AGGREGATOR}_random',
                f'encoder_attention_{DIM_AGGREGATOR}',
                f'encoder_summedNorms_{DIM_AGGREGATOR}',
                f'encoder_valueZeroing-cosine_{DIM_AGGREGATOR}',
                ]
if is_encoder_decoder:
                METHOD_NAMES.extend([
                f'decoder_attention_{DIM_AGGREGATOR}_random',
                f'decoder_summedNorms_{DIM_AGGREGATOR}_random',
                f'decoder_valueZeroing-cosine_{DIM_AGGREGATOR}_random',
                f'decoder_attention_{DIM_AGGREGATOR}',
                f'decoder_summedNorms_{DIM_AGGREGATOR}',
                f'decoder_valueZeroing-cosine_{DIM_AGGREGATOR}',
                f'cross_attention_{DIM_AGGREGATOR}_random',
                f'cross_summedNorms_{DIM_AGGREGATOR}_random',
                f'cross_valueZeroing-cosine_{DIM_AGGREGATOR}_random',
                f'cross_attention_{DIM_AGGREGATOR}',
                f'cross_summedNorms_{DIM_AGGREGATOR}',
                f'cross_valueZeroing-cosine_{DIM_AGGREGATOR}',
                ])
    
# Load processor 
processor = WhisperProcessor.from_pretrained(MODEL_PATH[MODEL_NAME], task='transcribe', language='french') if is_encoder_decoder else Wav2Vec2Processor.from_pretrained(MODEL_PATH[MODEL_NAME])

# Load scores
SCORES = {}
for method_name in METHOD_NAMES:
    with open(f'{SCORES_PATH}{method_name}.pkl', 'rb') as fp:
        SCORES[method_name] = pickle.load(fp)
num_layers = NUM_LAYERS[MODEL_NAME]

# load data
annot_data = load_from_disk(f"{ANNOTATED_DATA_PATH}{TEMPLATE}")
annot_data = annot_data.add_column("ex", list(range(len(annot_data))))

unique_org_id = [i for i in annot_data['org_id'] if annot_data['org_id'].count(i) == 1]
unique_ids = [i for i, x in enumerate(annot_data['org_id']) if x in unique_org_id]
annot_data = annot_data.select(unique_ids)
# 
# taking care of third template where there are two set of cue and targets 
D = annot_data.select(np.where(np.array(annot_data['template']) == "det_noun_verb")[0])
for x in D:
    temp = x
    temp['cue_indices']['enc'] = x['cue_indices']['enc']  + x['target_indices']['enc'] 
    temp['target_indices']['enc']  = x['target_indices_2']['enc'] 
    if is_encoder_decoder:
        temp['cue_indices']['dec'][MODEL_NAME] = x['cue_indices']['dec'][MODEL_NAME]  + x['target_indices']['dec'][MODEL_NAME]
        temp['target_indices']['dec'][MODEL_NAME]  = x['target_indices_2']['dec'][MODEL_NAME]
    annot_data = annot_data.add_item(temp)

num_examples = len(annot_data) 

# run
cue_alignments_dot_product = {}
progress_bar = tqdm(range(len(METHOD_NAMES)))
for method_name in METHOD_NAMES:
    dp = np.zeros((num_layers, num_examples)) 
    for ex in range(num_examples):
        score_maps = SCORES[method_name][annot_data[ex]['ex']]
        if len(score_maps.shape) == 4:
            score_maps = score_maps.mean(1) # mean over heads
        
        if method_name.split('_')[0] == 'cross':
            target_indices = np.array(annot_data[ex]['target_indices']['dec'][MODEL_NAME])
            cue_indices = np.array(annot_data[ex]['cue_indices']['enc'])
        elif method_name.split('_')[0] == 'decoder':
            target_indices = np.array(annot_data[ex]['target_indices']['dec'][MODEL_NAME]) # bos is already ignored in saved generated ids
            cue_indices = np.array(annot_data[ex]['cue_indices']['dec'][MODEL_NAME]) + 1 # due to bos token in input decoder
        elif method_name.split('_')[0] == 'encoder':
            target_indices = np.array(annot_data[ex]['target_indices']['enc'])
            cue_indices = np.array(annot_data[ex]['cue_indices']['enc'])
        else:
            print("something is wrong in score's name!")
            exit()
    
        if SKIP_SPECIAL_TOKENS:
            if method_name.split('_')[0] == 'cross':
                score_maps = score_maps[:, 3:]
                target_indices -= 3
            elif method_name.split('_')[0] == 'decoder':
                score_maps = score_maps[:, 3:, 3:]
                target_indices -= 3
                cue_indices -= 3

            score_maps = score_maps / np.sum(score_maps, axis=-1, keepdims=True)

        # binary cue vector
        cue_vector = np.zeros(score_maps.shape[-1])
        cue_vector[cue_indices] = 1
        
        for layer in range(num_layers):
            # loop over scores from viewpoint of each splitted tokens belongs to the generated target word
            dots = []
            for t in target_indices:
                s = score_maps[layer, t]
                dots.append(np.dot(s, cue_vector))
            dp[layer, ex] = np.mean(dots)
    # mean, max, or first
    cue_alignments_dot_product[method_name] = dp.mean(axis=-1) 

    progress_bar.update(1)

# save evals
postfix = "_" + DIM_AGGREGATOR
if SKIP_SPECIAL_TOKENS:
    postfix = postfix + "_skipped" 

with open(f'{SAVE_EVAL_PATH}dp{postfix}.pkl', 'wb') as f:
    pickle.dump(cue_alignments_dot_product, f)

