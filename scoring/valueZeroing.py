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

METRIC = 'cosine'
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
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from datasets import load_from_disk
from transformers import WhisperProcessor, Wav2Vec2Processor, AutoConfig
from modeling.customized_modeling_whisper import WhisperForConditionalGeneration
from modeling.customized_modeling_wav2vec2 import Wav2Vec2ForCTC
from utils import MODEL_PATH, get_encoder_word_boundaries


if not os.path.exists(SAVE_SCORES_PATH):
    os.makedirs(SAVE_SCORES_PATH)

DISTANCE_FUNC = {'cosine': cosine_distances,
                 'euclidean': euclidean_distances
                }

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
all_enc_value_zeroing = []
if is_encoder_decoder:
    all_dec_value_zeroing = []
    all_cross_value_zeroing = []

progress_bar = tqdm(range(num_examples))
for ex in range(num_examples):
    # original inference
    with torch.no_grad():
        if is_encoder_decoder:
            original_outputs = model(all_input_features[ex].input_features, 
                            decoder_input_ids=all_decoder_input_ids[ex],
                            output_hidden_states=True,
                            return_dict=True)
        else:
            original_outputs = model(all_input_features[ex].input_values, 
                            attention_mask=all_input_features[ex].attention_mask,
                            output_hidden_states=True,
                            return_dict=True)

    original_enc_hidden_states = torch.stack(original_outputs[f'{prefix}hidden_states'])
    if is_encoder_decoder:
        original_dec_hidden_states = torch.stack(original_outputs['decoder_hidden_states'])

    # prepare attention mask
    if is_encoder_decoder:
        # causal attention mask
        attention_mask = model.model.decoder._prepare_decoder_attention_mask(
            attention_mask=None, input_shape=all_decoder_input_ids[ex].size(), inputs_embeds=model.model.decoder.embed_tokens(all_decoder_input_ids[ex]), past_key_values_length=0) 
    else:
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask = all_input_features[ex].attention_mask
        extract_features = model.wav2vec2.feature_extractor(all_input_features[ex].input_values)
        extract_features = extract_features.transpose(1, 2)
        attention_mask = model.wav2vec2._get_feature_vector_attention_mask(
                    extract_features.shape[1], attention_mask, add_adapter=False
                )
        # extend attention_mask
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=original_enc_hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(original_enc_hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    # ready to form and extract attentions
    # whisper audio inputs are always padded to max length (30 seconds) since whisper doesn't support attetnion mask in encoder part
    total_audio_time = 30.0 if is_encoder_decoder else len(annot_data[ex]['audio']['array']) / annot_data[ex]['audio']['sampling_rate'] 
    encoder_aligned_length = len(annot_data[ex]['alignment']['intervals'])
    total_enc_dimensions = original_enc_hidden_states.shape[2]
    num_enc_layers = model.config.encoder_layers if is_encoder_decoder else model.config.num_hidden_layers
    if is_encoder_decoder:
        decoder_length = all_decoder_input_ids[ex].shape[-1]


    # compute scores: encoder
    vz_enc_matrix = np.zeros(shape=(num_enc_layers, encoder_aligned_length, encoder_aligned_length))
    for l, encoder_layer in enumerate(model.model.encoder.layers if is_encoder_decoder else model.wav2vec2.encoder.layers):
        for t in range(encoder_aligned_length):
            start_j, end_j = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][t]['start'], annot_data[ex]['alignment']['intervals'][t]['end'], total_enc_dimensions, total_audio_time)
            with torch.no_grad():
                layer_outputs = encoder_layer(
                                    hidden_states=original_enc_hidden_states[l], 
                                    attention_mask=None if is_encoder_decoder else attention_mask,
                                    value_zeroing=True,
                                    value_zeroing_index=(start_j, end_j),
                                    value_zeroing_head="all",
                                    )
            
            alternative_hidden_states = layer_outputs[0]
            # last layer is followed by a layer normalization
            if l == num_enc_layers - 1: 
                alternative_hidden_states = model.model.encoder.layer_norm(alternative_hidden_states) if is_encoder_decoder else model.wav2vec2.encoder.layer_norm(alternative_hidden_states)
             
            x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
            y = original_enc_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
            distances = DISTANCE_FUNC[METRIC](x, y).diagonal()
            for i in range(encoder_aligned_length):
                start_i, end_i = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][i]['start'], annot_data[ex]['alignment']['intervals'][i]['end'], total_enc_dimensions, total_audio_time)
                vz_enc_matrix[l, i, t] = distances[start_i:end_i+1].max() if DIM_AGGREGATOR == "max" else distances[start_i:end_i+1].mean()

    vz_enc_matrix = vz_enc_matrix / np.sum(vz_enc_matrix, axis=-1, keepdims=True)
    all_enc_value_zeroing.append(vz_enc_matrix)

    if is_encoder_decoder:

        # compute scores: decoder
        vz_dec_matrix = np.zeros(shape=(model.config.decoder_layers, decoder_length, decoder_length))
        for l, decoder_layer in enumerate(model.model.decoder.layers):
            for t in range(decoder_length):
                with torch.no_grad():
                    layer_outputs = decoder_layer(
                                hidden_states=original_dec_hidden_states[l],
                                attention_mask=attention_mask,
                                encoder_hidden_states=original_enc_hidden_states[-1],
                                past_key_value=None,
                                value_zeroing="decoder",
                                value_zeroing_index=t,
                                value_zeroing_head="all",
                                )
    
                alternative_hidden_states = layer_outputs[0]
                if l == model.config.decoder_layers - 1: # last layer in whisper is followed by a layer normalization
                    alternative_hidden_states = model.model.decoder.layer_norm(alternative_hidden_states)
                
                x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                y = original_dec_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                distances = DISTANCE_FUNC[METRIC](x, y).diagonal()
                # only tokens after t is considerd to see how much they are changed after zeroing t. tokens < t have not seen t yet!
                vz_dec_matrix[l, t:, t] = distances[t:]
        
        sums = np.sum(vz_dec_matrix, axis=-1, keepdims=True)
        mask = np.all(sums == 0, axis=-1, keepdims=True)
        vz_dec_matrix = np.divide(vz_dec_matrix, sums, out=np.zeros_like(vz_dec_matrix), where=~mask)
        all_dec_value_zeroing.append(vz_dec_matrix)

        # compute scores: cross
        vz_cross_matrix = np.zeros(shape=(model.config.decoder_layers, decoder_length, encoder_aligned_length))
        for l, decoder_layer in enumerate(model.model.decoder.layers):
            for t in range(encoder_aligned_length):
                start_j, end_j = get_encoder_word_boundaries(annot_data[ex]['alignment']['intervals'][t]['start'], annot_data[ex]['alignment']['intervals'][t]['end'], total_enc_dimensions, total_audio_time)
                with torch.no_grad():
                    layer_outputs = decoder_layer(
                                hidden_states=original_dec_hidden_states[l],
                                attention_mask=attention_mask,
                                encoder_hidden_states=original_enc_hidden_states[-1],
                                past_key_value=None,
                                value_zeroing="cross",
                                value_zeroing_index=(start_j, end_j),
                                value_zeroing_head="all",
                                )
                
                alternative_hidden_states = layer_outputs[0]
                if l == model.config.decoder_layers - 1: # last layer in whisper is followed by a layer normalization
                    alternative_hidden_states = model.model.decoder.layer_norm(alternative_hidden_states)

                x = alternative_hidden_states.squeeze(0).detach().cpu().numpy()
                y = original_dec_hidden_states[l+1].squeeze(0).detach().cpu().numpy()
                distances = DISTANCE_FUNC[METRIC](x, y).diagonal()
                vz_cross_matrix[l, :, t] = distances
        sums = np.sum(vz_cross_matrix, axis=-1, keepdims=True)
        mask = np.all(sums == 0, axis=-1, keepdims=True)
        vz_cross_matrix = np.divide(vz_cross_matrix, sums, out=np.zeros_like(vz_cross_matrix), where=~mask)
        all_cross_value_zeroing.append(vz_cross_matrix)
        

    progress_bar.update(1)

# Save 
postfix = "_" + DIM_AGGREGATOR
if RANDOM:
    postfix = postfix + "_random" 

with open(f'{SAVE_SCORES_PATH}encoder_valueZeroing-{METRIC}{postfix}.pkl', 'wb') as f:
    pickle.dump(all_enc_value_zeroing, f)
if is_encoder_decoder:
    with open(f'{SAVE_SCORES_PATH}decoder_valueZeroing-{METRIC}{postfix}.pkl', 'wb') as f:
        pickle.dump(all_dec_value_zeroing, f)
    with open(f'{SAVE_SCORES_PATH}cross_valueZeroing-{METRIC}{postfix}.pkl', 'wb') as f:
        pickle.dump(all_cross_value_zeroing, f)