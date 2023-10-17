import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--RANDOM", action='store_true')
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
RANDOM = args.RANDOM

# MODEL_NAME = "whisper-medium" 
# RANDOM = False

TASK = "common_voice"
SPLIT = "test" 
TEMPLATE = "all"
SELECTED_GPU = 0
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"
GENERATED_IDS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/{MODEL_NAME}/"
SAVE_OUTPUT_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/representations/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, Wav2Vec2Processor, AutoConfig
from modeling.customized_modeling_whisper import WhisperForConditionalGeneration
from modeling.customized_modeling_wav2vec2 import Wav2Vec2ForCTC
from utils import MODEL_PATH, get_encoder_word_boundaries

if not os.path.exists(SAVE_OUTPUT_PATH):
    os.makedirs(SAVE_OUTPUT_PATH)

### GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()

is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"

# load annotated data
annot_data = load_from_disk(f"{ANNOTATED_DATA_PATH}{TEMPLATE}")
annot_data = annot_data.add_column("ex", list(range(len(annot_data))))
unique_org_id = [i for i in annot_data['org_id'] if annot_data['org_id'].count(i) == 1]
unique_ids = [i for i, x in enumerate(annot_data['org_id']) if x in unique_org_id]
annot_data = annot_data.select(unique_ids)

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

# Load processor and model
processor = WhisperProcessor.from_pretrained(MODEL_PATH[MODEL_NAME], task='transcribe', language='french') if is_encoder_decoder else Wav2Vec2Processor.from_pretrained(MODEL_PATH[MODEL_NAME])
if RANDOM:
    model = WhisperForConditionalGeneration(AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME])) if is_encoder_decoder else Wav2Vec2ForCTC(AutoConfig.from_pretrained(MODEL_PATH[MODEL_NAME]))
else:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH[MODEL_NAME]) if is_encoder_decoder else Wav2Vec2ForCTC.from_pretrained(MODEL_PATH[MODEL_NAME])
if is_encoder_decoder:
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "fr", task = "transcribe")
model.to(device)
model.eval()

# Load generated token ids
if is_encoder_decoder:
    with open(f'{GENERATED_IDS_PATH}generated_ids.pkl', 'rb') as fp:
        generated_ids = pickle.load(fp)
    all_decoder_input_ids = []
    for ex in range(num_examples):
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id] + generated_ids[annot_data[ex]['org_id']][:-1].tolist()])
        all_decoder_input_ids.append(decoder_input_ids.to(device))

# input to features
all_input_features = []
for ex in range(num_examples):
    audio = annot_data[ex]["audio"]["array"]
    sampling_rate = annot_data[ex]['audio']['sampling_rate']
    if is_encoder_decoder:
        features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").to(device) 
    else:
        features = processor(audio, sampling_rate=sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt").to(device) 
    all_input_features.append(features)

prefix = "encoder_" if is_encoder_decoder else ""

# run
all_enc_target_representations = []
if is_encoder_decoder:
    all_dec_target_representations = []
progress_bar = tqdm(range(num_examples))
for ex in range(num_examples):
    # inference
    with torch.no_grad():
        if is_encoder_decoder:
            outputs = model(all_input_features[ex].input_features, 
                            decoder_input_ids=all_decoder_input_ids[ex],
                            output_hidden_states=True,
                            return_dict=True)
        else:
            outputs = model(all_input_features[ex].input_values, 
                            attention_mask=all_input_features[ex].attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
    
    enc_hidden_states = torch.stack(outputs[f'{prefix}hidden_states']).squeeze(1).detach().cpu().numpy()
    if is_encoder_decoder:
        dec_hidden_states = torch.stack(outputs['decoder_hidden_states']).squeeze(1).detach().cpu().numpy()

    # find target index/frames
    # whisper audio inputs are always padded to max length (30 seconds) since whisper doesn't support attetnion mask in encoder part
    total_audio_time = 30.0 if is_encoder_decoder else len(annot_data[ex]['audio']['array']) / annot_data[ex]['audio']['sampling_rate'] 
    total_enc_dimensions = enc_hidden_states.shape[1] # number of frames
    enc_target_index = annot_data[ex]['target_indices']['enc']
    if is_encoder_decoder:
        dec_target_index = annot_data[ex]['target_indices']['dec'][MODEL_NAME]
    
    enc_target_frames = []
    for c in enc_target_index:
        start, end = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][c]['start'], annot_data[ex]['alignment']['intervals'][c]['end'], total_enc_dimensions, total_audio_time)
        enc_target_frames.extend(list(range(start, end+1)))
    all_enc_target_representations.append(enc_hidden_states[1:, enc_target_frames].mean(1))
    if is_encoder_decoder:
        all_dec_target_representations.append(dec_hidden_states[1:, dec_target_index].mean(1))
    
    progress_bar.update(1)


# Save 
postfix = ""
if RANDOM:
    postfix += "_random"
with open(f'{SAVE_OUTPUT_PATH}enc_target_representations{postfix}.pkl', 'wb') as f:
    pickle.dump(all_enc_target_representations, f)
if is_encoder_decoder:
    with open(f'{SAVE_OUTPUT_PATH}dec_target_representations{postfix}.pkl', 'wb') as f:
        pickle.dump(all_dec_target_representations, f)
