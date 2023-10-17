import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
parser.add_argument("--DIM_AGGREGATOR")
parser.add_argument("--RANDOM", action='store_true')
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
DIM_AGGREGATOR = args.DIM_AGGREGATOR
RANDOM = args.RANDOM

# MODEL_NAME = "whisper-medium" 
# RANDOM = False
# DIM_AGGREGATOR = "mean"

TASK = "common_voice"
SPLIT = "test" 
TEMPLATE = "all"
SELECTED_GPU = 0
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"
GENERATED_IDS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/{MODEL_NAME}/"
SAVE_SCORES_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/scores/{TASK}/{SPLIT}/{TEMPLATE}/{MODEL_NAME}/"

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

if not os.path.exists(SAVE_SCORES_PATH):
    os.makedirs(SAVE_SCORES_PATH)

### GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    # exit()

# load annotated data
annot_data = load_from_disk(f"{ANNOTATED_DATA_PATH}{TEMPLATE}")
num_examples = len(annot_data)

# Load processor and model
is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"
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
    if is_encoder_decoder:
        features = processor(annot_data[ex]["audio"]["array"], sampling_rate=annot_data[ex]['audio']['sampling_rate'], return_tensors="pt").to(device) 
    else:
        features = processor(annot_data[ex]["audio"]["array"], sampling_rate=annot_data[ex]['audio']['sampling_rate'], padding=True, return_attention_mask=True, return_tensors="pt").to(device) 
    all_input_features.append(features)

prefix = "encoder_" if is_encoder_decoder else ""

# run
all_enc_attentions = []
all_enc_norms = []
all_enc_summed_norms = []
if is_encoder_decoder:
    all_dec_attentions = []
    all_dec_norms = []
    all_dec_summed_norms = []
    all_cross_attentions = []
    all_cross_norms = []
    all_cross_summed_norms = []

progress_bar = tqdm(range(num_examples))
for ex in range(num_examples):
    # inference
    with torch.no_grad():
        if is_encoder_decoder:
            _, max_frame = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][-1]['start'], annot_data[ex]['alignment']['intervals'][-1]['end'], 1500, 30.0)
            outputs = model(all_input_features[ex].input_features, 
                            decoder_input_ids=all_decoder_input_ids[ex],
                            output_attentions=True,
                            output_norms=True,
                            max_frame=max_frame+1,
                            return_dict=True)
        else:
            outputs = model(all_input_features[ex].input_values, 
                            attention_mask=all_input_features[ex].attention_mask,
                            output_attentions=True,
                            output_norms=True, # due to limited budget
                            return_dict=True)

    # ready to form and extract attentions
    # whisper audio inputs are always padded to max length (30 seconds) since whisper doesn't support attetnion mask in encoder part
    total_audio_time = 30.0 if is_encoder_decoder else len(annot_data[ex]['audio']['array']) / annot_data[ex]['audio']['sampling_rate'] 
    encoder_aligned_length = len(annot_data[ex]['alignment']['intervals'])
    num_enc_layers = model.config.encoder_layers if is_encoder_decoder else model.config.num_hidden_layers
    num_enc_heads = model.config.encoder_attention_heads if is_encoder_decoder else model.config.num_attention_heads
    if is_encoder_decoder:
        decoder_length = all_decoder_input_ids[ex].shape[-1]


    ## extract from encoder 
    encoder_attentions = np.zeros(shape=(num_enc_layers, num_enc_heads, encoder_aligned_length, encoder_aligned_length))
    encoder_norms = np.zeros(shape=(num_enc_layers, num_enc_heads, encoder_aligned_length, encoder_aligned_length))
    encoder_summed_norms = np.zeros(shape=(num_enc_layers, encoder_aligned_length, encoder_aligned_length))
    
    attn_scores = torch.stack(outputs[f'{prefix}attentions']).squeeze(1).detach().cpu().numpy()
    norm_scores = torch.stack(outputs[f'{prefix}norms']).squeeze(1).detach().cpu().numpy()
    summed_norm_scores = torch.stack(outputs[f'{prefix}summed_norms']).squeeze(1).detach().cpu().numpy()
    total_enc_dimensions = attn_scores.shape[-1] 
    for i in range(encoder_aligned_length):
        start_i, end_i = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][i]['start'], annot_data[ex]['alignment']['intervals'][i]['end'], total_enc_dimensions, total_audio_time)
        for j in range(encoder_aligned_length):
            start_j, end_j = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][j]['start'], annot_data[ex]['alignment']['intervals'][j]['end'], total_enc_dimensions, total_audio_time)
            encoder_attentions[:, :, i, j] = attn_scores[:, :, start_i:end_i+1, start_j:end_j+1].max(-1).max(-1) if DIM_AGGREGATOR == "max" else attn_scores[:, :, start_i:end_i+1, start_j:end_j+1].mean(-1).mean(-1)
            encoder_norms[:, :, i, j] = norm_scores[:, :, start_i:end_i+1, start_j:end_j+1].max(-1).max(-1) if DIM_AGGREGATOR == "max" else norm_scores[:, :, start_i:end_i+1, start_j:end_j+1].mean(-1).mean(-1)
            encoder_summed_norms[:, i, j] = summed_norm_scores[:, start_i:end_i+1, start_j:end_j+1].max(-1).max(-1) if DIM_AGGREGATOR == "max" else summed_norm_scores[:, start_i:end_i+1, start_j:end_j+1].mean(-1).mean(-1)

    # normalize
    sums = np.sum(encoder_attentions, axis=-1, keepdims=True)
    mask = np.all(sums == 0, axis=-1, keepdims=True)
    encoder_attentions = np.divide(encoder_attentions, sums, out=np.zeros_like(encoder_attentions), where=~mask)
    
    sums = np.sum(encoder_norms, axis=-1, keepdims=True)
    mask = np.all(sums == 0, axis=-1, keepdims=True)
    encoder_norms = np.divide(encoder_norms, sums, out=np.zeros_like(encoder_norms), where=~mask)
    
    sums = np.sum(encoder_summed_norms, axis=-1, keepdims=True)
    mask = np.all(sums == 0, axis=-1, keepdims=True)
    encoder_summed_norms = np.divide(encoder_summed_norms, sums, out=np.zeros_like(encoder_summed_norms), where=~mask)
    
    # append
    all_enc_attentions.append(encoder_attentions)
    all_enc_norms.append(encoder_norms)
    all_enc_summed_norms.append(encoder_summed_norms)

    if is_encoder_decoder:
        # extract from decoder
        decoder_attentions = torch.stack(outputs['decoder_attentions']).squeeze(1).detach().cpu().numpy()
        decoder_norms = torch.stack(outputs['decoder_norms']).squeeze(1).detach().cpu().numpy()
        decoder_summed_norms = torch.stack(outputs['decoder_summed_norms']).squeeze(1).detach().cpu().numpy()
        decoder_attentions = decoder_attentions / np.sum(decoder_attentions, axis=-1, keepdims=True)
        decoder_norms = decoder_norms / np.sum(decoder_norms, axis=-1, keepdims=True)
        decoder_summed_norms = decoder_summed_norms / np.sum(decoder_summed_norms, axis=-1, keepdims=True)
        all_dec_attentions.append(decoder_attentions)
        all_dec_norms.append(decoder_norms)
        all_dec_summed_norms.append(decoder_summed_norms)

        # extract cross attentions
        cross_attentions = np.zeros(shape=(model.config.decoder_layers, model.config.decoder_attention_heads, decoder_length, encoder_aligned_length))
        cross_norms = np.zeros(shape=(model.config.decoder_layers, model.config.decoder_attention_heads, decoder_length, encoder_aligned_length))
        cross_summed_norms = np.zeros(shape=(model.config.decoder_layers, decoder_length, encoder_aligned_length))
        attn_scores = torch.stack(outputs['cross_attentions']).squeeze(1).detach().cpu().numpy()
        norm_scores = torch.stack(outputs['cross_norms']).squeeze(1).detach().cpu().numpy()
        summed_norm_scores = torch.stack(outputs['cross_summed_norms']).squeeze(1).detach().cpu().numpy()
        for j in range(encoder_aligned_length):
            start_j, end_j = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][j]['start'], annot_data[ex]['alignment']['intervals'][j]['end'], total_enc_dimensions, total_audio_time)
            cross_attentions[:, :, :, j] = attn_scores[:, :, :, start_j:end_j+1].max(axis=-1) if DIM_AGGREGATOR == "max" else attn_scores[:, :, :, start_j:end_j+1].mean(axis=-1)
            cross_norms[:, :, :, j] = norm_scores[:, :, :, start_j:end_j+1].max(axis=-1) if DIM_AGGREGATOR == "max" else norm_scores[:, :, :, start_j:end_j+1].mean(axis=-1)
            cross_summed_norms[:, :, j] = summed_norm_scores[:, :, start_j:end_j+1].max(axis=-1) if DIM_AGGREGATOR == "max" else summed_norm_scores[:, :, start_j:end_j+1].mean(axis=-1)
        cross_attentions = cross_attentions / np.sum(cross_attentions, axis=-1, keepdims=True)
        cross_norms = cross_norms / np.sum(cross_norms, axis=-1, keepdims=True)
        cross_summed_norms = cross_summed_norms / np.sum(cross_summed_norms, axis=-1, keepdims=True)
        all_cross_attentions.append(cross_attentions)
        all_cross_norms.append(cross_norms)
        all_cross_summed_norms.append(cross_summed_norms)

    
    progress_bar.update(1)


# Save
postfix = "_" + DIM_AGGREGATOR
if RANDOM:
    postfix = postfix + "_random" 

# encoder
with open(f'{SAVE_SCORES_PATH}encoder_attention{postfix}.pkl', 'wb') as f:
    pickle.dump(all_enc_attentions, f)
with open(f'{SAVE_SCORES_PATH}encoder_norms{postfix}.pkl', 'wb') as f:
    pickle.dump(all_enc_norms, f)
with open(f'{SAVE_SCORES_PATH}encoder_summedNorms{postfix}.pkl', 'wb') as f:
    pickle.dump(all_enc_summed_norms, f)

if is_encoder_decoder:
    # decoder
    with open(f'{SAVE_SCORES_PATH}decoder_attention{postfix}.pkl', 'wb') as f:
        pickle.dump(all_dec_attentions, f)
    with open(f'{SAVE_SCORES_PATH}decoder_norms{postfix}.pkl', 'wb') as f:
        pickle.dump(all_dec_norms, f)
    with open(f'{SAVE_SCORES_PATH}decoder_summedNorms{postfix}.pkl', 'wb') as f:
        pickle.dump(all_dec_summed_norms, f)
    # cross
    with open(f'{SAVE_SCORES_PATH}cross_attention{postfix}.pkl', 'wb') as f:
        pickle.dump(all_cross_attentions, f)
    with open(f'{SAVE_SCORES_PATH}cross_norms{postfix}.pkl', 'wb') as f:
        pickle.dump(all_cross_norms, f)
    with open(f'{SAVE_SCORES_PATH}cross_summedNorms{postfix}.pkl', 'wb') as f:
        pickle.dump(all_cross_summed_norms, f)

