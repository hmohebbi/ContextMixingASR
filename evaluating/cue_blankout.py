import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--PROB_AGGREGATOR")
parser.add_argument("--TARGET", action='store_true')
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
PROB_AGGREGATOR = args.PROB_AGGREGATOR
TARGET = args.TARGET

# MODEL_NAME = "whisper-medium"
# PROB_AGGREGATOR = "mean"
# TARGET = False

TEMPLATE = "all" 
TASK = "common_voice"
SPLIT = "test" 
SELECTED_GPU = 0
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"
GENERATED_IDS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/{MODEL_NAME}/"
SAVE_EVAL_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/evals/faithfulness/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"
SAVE_FIGURES_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/figures/cue_alignment/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import IPython.display as ipd
from datasets import load_from_disk
from transformers import WhisperProcessor, Wav2Vec2Processor
from modeling.customized_modeling_whisper import WhisperForConditionalGeneration
from modeling.customized_modeling_wav2vec2 import Wav2Vec2ForCTC
from utils import MODEL_PATH, get_encoder_word_boundaries

if not os.path.exists(SAVE_EVAL_PATH):
    os.makedirs(SAVE_EVAL_PATH)

### GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()

# Load generated token ids
is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"
if is_encoder_decoder:
    with open(f'{GENERATED_IDS_PATH}generated_ids.pkl', 'rb') as fp:
        generated_ids = pickle.load(fp)

# load annotated data
annot_data = load_from_disk(f"{ANNOTATED_DATA_PATH}{TEMPLATE}")
unique_org_id = [i for i in annot_data['org_id'] if annot_data['org_id'].count(i) == 1]
unique_ids = [i for i, x in enumerate(annot_data['org_id']) if x in unique_org_id]
annot_data = annot_data.select(unique_ids)
num_examples = len(annot_data)

# Load processor and model
processor = WhisperProcessor.from_pretrained(MODEL_PATH[MODEL_NAME], task='transcribe', language='french') if is_encoder_decoder else Wav2Vec2Processor.from_pretrained(MODEL_PATH[MODEL_NAME])
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH[MODEL_NAME]) if is_encoder_decoder else Wav2Vec2ForCTC.from_pretrained(MODEL_PATH[MODEL_NAME])
if is_encoder_decoder:
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "fr", task = "transcribe")
model.to(device)
model.eval()

# run
all_silencing_drop = []
if is_encoder_decoder:
    all_blankout_drop = []
    all_both_drop = []
progress_bar = tqdm(range(num_examples))
for ex in range(num_examples):
    sampling_rate = annot_data[ex]['audio']['sampling_rate']
    total_audio_time = len(annot_data[ex]['audio']['array']) / sampling_rate # actual audio time before processing
    total_audio_frames = len(annot_data[ex]["audio"]["array"])
    if TARGET:
        enc_cue_indices = np.array(annot_data[ex]['target_indices']['enc'])
    else:
        enc_cue_indices = annot_data[ex]['cue_indices']['enc']
    
    if is_encoder_decoder:
        
        # original run
        org_input_features = processor(annot_data[ex]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device) 
        org_decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id] + generated_ids[annot_data[ex]['org_id']][:-1].tolist()]).to(device) 
        with torch.no_grad():
            org_outputs = model(org_input_features, 
                                decoder_input_ids=org_decoder_input_ids,
                                return_dict=True)
        
        org_probs = torch.nn.functional.softmax(org_outputs['logits'].squeeze(0), dim=-1)
        org_target_preds = torch.argmax(org_probs, dim=-1).detach().cpu().numpy()
        org_probs = org_probs[torch.arange(org_probs.size(0)), org_target_preds][annot_data[ex]['target_indices']['dec'][MODEL_NAME]].detach().cpu().numpy()
        
        # prepare purturbed inputs
        # enc
        alternative_audio = np.array(annot_data[ex]["audio"]["array"])
        silenced_indices = []
        for index in enc_cue_indices:
            start_c, end_c = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][index]['start'], annot_data[ex]['alignment']['intervals'][index]['end'], total_audio_frames, total_audio_time)
            silenced_indices.extend(list(range(start_c, end_c+1)))
        alternative_audio[silenced_indices] = 0.0
        # print(annot_data[ex]['alignment']['intervals'][enc_cue_index]['word'])
        # ipd.Audio(alternative_audio, rate=sampling_rate)
        alternative_input_features = processor(alternative_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device) 
        # dec
        alternative_decoder_input_ids = org_decoder_input_ids.detach().clone()
        if TARGET:
            cue_input_position = np.array(annot_data[ex]['target_indices']['dec'][MODEL_NAME]) # without +1 is prev token to target
        else:
            cue_input_position = np.array(annot_data[ex]['cue_indices']['dec'][MODEL_NAME]) + 1 # indexing was based on generated output tokens, for input we have a prepended BOS
        alternative_decoder_input_ids[:, cue_input_position] = processor.tokenizer.unk_token_id
        
        # unify different perturbed inputs as a batch
        # [encoder-silencing, decoder-blanking_out, both]
        batch_input_features = torch.cat((alternative_input_features, org_input_features, alternative_input_features), 0)
        batch_decoder_input_ids = torch.cat((org_decoder_input_ids, alternative_decoder_input_ids, alternative_decoder_input_ids), 0)

        with torch.no_grad():
            alternative_outputs = model(batch_input_features, 
                                        decoder_input_ids=batch_decoder_input_ids,
                                        return_dict=True)
        
        alternative_probs = torch.nn.functional.softmax(alternative_outputs['logits'], dim=-1)
        alternative_probs = alternative_probs[:, torch.arange(alternative_probs.size(1)), org_target_preds]
        alternative_probs = alternative_probs[:, annot_data[ex]['target_indices']['dec'][MODEL_NAME]].detach().cpu().numpy()
        
        # diff prob
        diff_prob = (org_probs - alternative_probs).max(1) if PROB_AGGREGATOR == "max" else (org_probs - alternative_probs).mean(1)
        
        all_silencing_drop.append(diff_prob[0].item())
        all_blankout_drop.append(diff_prob[1].item())
        all_both_drop.append(diff_prob[2].item())


    else:
        inputs = processor(annot_data[ex]["audio"]["array"], sampling_rate=sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt").to(device) 
        with torch.no_grad():
            org_outputs = model(inputs.input_values, 
                                attention_mask=inputs.attention_mask,
                                output_hidden_states=True,
                                return_dict=True)
        
        
        total_enc_frames = torch.stack(org_outputs['hidden_states']).shape[-2]
        enc_indices = []
        for index in enc_cue_indices:
            start_enc_c, end_enc_c = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][index]['start'], annot_data[ex]['alignment']['intervals'][index]['end'], total_enc_frames, total_audio_time)
            enc_indices.extend(list(range(start_enc_c, end_enc_c+1)))
        org_probs = torch.nn.functional.softmax(org_outputs['logits'].squeeze(0), dim=-1)
        org_target_preds = torch.argmax(org_probs, dim=-1).detach().cpu().numpy()
        org_probs = org_probs[torch.arange(org_probs.size(0)), org_target_preds][enc_indices]
        org_probs = org_probs.detach().cpu().numpy()
        
        # prepare purturbed inputs
        # enc
        alternative_audio = np.array(annot_data[ex]["audio"]["array"])
        silenced_indices = []
        for index in enc_cue_indices:
            start_c, end_c = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][index]['start'], annot_data[ex]['alignment']['intervals'][index]['end'], total_audio_frames, total_audio_time)
            silenced_indices.extend(list(range(start_c, end_c+1)))
        alternative_audio[silenced_indices] = 0.0
        # print(annot_data[ex]['alignment']['intervals'][enc_cue_index]['word'])
        # ipd.Audio(alternative_audio, rate=sampling_rate)
        alternative_inputs = processor(alternative_audio, sampling_rate=sampling_rate, padding=True, return_attention_mask=True, return_tensors="pt").to(device) 
        
        with torch.no_grad():
            alternative_outputs = model(alternative_inputs.input_values, 
                                        attention_mask=alternative_inputs.attention_mask,
                                        return_dict=True)
        
        alternative_probs = torch.nn.functional.softmax(alternative_outputs['logits'].squeeze(0), dim=-1)
        alternative_probs = alternative_probs[torch.arange(alternative_probs.size(0)), org_target_preds][enc_indices]
        alternative_probs = alternative_probs.detach().cpu().numpy()

        # diff prob
        diff_prob = (org_probs - alternative_probs).max() if PROB_AGGREGATOR == "max" else (org_probs - alternative_probs).mean()
        all_silencing_drop.append(diff_prob.item())


    progress_bar.update(1)


# save
postfix = "_" + PROB_AGGREGATOR
if TARGET:
    postfix += "_target-1"
with open(f'{SAVE_EVAL_PATH}cue_silencing{postfix}.pkl', 'wb') as f:
    pickle.dump(all_silencing_drop, f)
if is_encoder_decoder:
    with open(f'{SAVE_EVAL_PATH}cue_blankout{postfix}.pkl', 'wb') as f:
        pickle.dump(all_blankout_drop, f)
    with open(f'{SAVE_EVAL_PATH}cue_both{postfix}.pkl', 'wb') as f:
        pickle.dump(all_both_drop, f)



