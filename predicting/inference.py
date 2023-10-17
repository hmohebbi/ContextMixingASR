import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_NAME")
args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME

# MODEL_NAME = "whisper-medium"

TASK = "common_voice"
SPLIT = "test" 
SELECTED_GPU = 0
SAVE_OUTPUT_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/{MODEL_NAME}/"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, Wav2Vec2Processor
from modeling.customized_modeling_whisper import WhisperForConditionalGeneration
from modeling.customized_modeling_wav2vec2 import Wav2Vec2ForCTC
from evaluate import load
from utils import MODEL_PATH, DATA_KEY, TEXT_KEY

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


# load data
data = load_dataset(DATA_KEY[TASK], 'fr', split=SPLIT, verification_mode="all_checks")
data = data.cast_column("audio", Audio(sampling_rate=16_000))

# Load processor and model
is_encoder_decoder = MODEL_NAME.split('-')[0] == "whisper"
processor = WhisperProcessor.from_pretrained(MODEL_PATH[MODEL_NAME], task='transcribe', language='french') if MODEL_NAME.split('-')[0] == "whisper" else Wav2Vec2Processor.from_pretrained(MODEL_PATH[MODEL_NAME])
whisper_processor = WhisperProcessor.from_pretrained(MODEL_PATH["whisper-small"], task='transcribe', language='french')
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH[MODEL_NAME]) if MODEL_NAME.split('-')[0] == "whisper" else Wav2Vec2ForCTC.from_pretrained(MODEL_PATH[MODEL_NAME])
if is_encoder_decoder: 
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "fr", task = "transcribe")
model.to(device)
model.eval()

# evaluation metric
wer_metric = load("wer")

# run
all_generated_ids = []
all_wers = []
progress_bar = tqdm(range(len(data)))
for ex in range(len(data)):
    # convert to input features
    inputs = processor(data[ex]["audio"]["array"], sampling_rate=data[ex]['audio']['sampling_rate'], return_tensors="pt")
    # inference
    with torch.no_grad():
        if is_encoder_decoder:
            outputs = model.generate(inputs=inputs.input_features.to(device), output_scores=True, return_dict_in_generate=True)
        else:
            outputs = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device))

    if is_encoder_decoder:
        generated_ids = outputs['sequences'].squeeze(0).detach().cpu().numpy()[1:] # discard prepended id: <|startoftranscript|>
        # <|fr|> <|transcribe|> <|notimestamps|> ids <|endoftext|>
    else:
        generated_ids = torch.argmax(outputs['logits'], dim=-1).squeeze(0).detach().cpu().numpy()
    
    # wer
    reference = data[ex][TEXT_KEY[TASK]]
    transcription = processor.decode(generated_ids)
    reference = whisper_processor.tokenizer._normalize(reference)
    transcription = whisper_processor.tokenizer._normalize(transcription)
    wer = 100 * wer_metric.compute(references=[reference], predictions=[transcription])

    all_generated_ids.append(generated_ids)
    all_wers.append(wer)

    progress_bar.update(1)


# Save 
with open(f'{SAVE_OUTPUT_PATH}generated_ids.pkl', 'wb') as f:
    pickle.dump(all_generated_ids, f)
with open(f'{SAVE_OUTPUT_PATH}wers.pkl', 'wb') as f:
    pickle.dump(all_wers, f)
