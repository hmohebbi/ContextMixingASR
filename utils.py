import numpy as np

det_words = ['le', 'la', 'les', 'un', 'une', 'des'] 
irregular_nouns = ['oeil', 'yeux', 'aïeul', 'aïeux', 'ciel', 'cieux', 'vieil', 'vieux']

MODEL_PATH = {
    # 'whisper-tiny': 'openai/whisper-tiny',
    'whisper-base': 'openai/whisper-base',
    'whisper-small': 'openai/whisper-small',
    'whisper-medium': 'openai/whisper-medium',
    'wav2vec2-large-xlsr-53-french': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
    'asr-wav2vec2-french': 'bhuang/asr-wav2vec2-french',
}

NUM_LAYERS = {
    # 'whisper-tiny': 4,
    'whisper-base': 6,
    'whisper-small': 12,
    'whisper-medium': 24,
    'wav2vec2-large-xlsr-53-french': 24,
    'asr-wav2vec2-french': 24,
}

DATA_KEY = {
    "common_voice": "mozilla-foundation/common_voice_11_0",
}
TEXT_KEY = {
    'common_voice': 'sentence',
}

def get_encoder_word_boundaries(start, end, total_enc_frame, total_audio_time):
    start = total_enc_frame * start / total_audio_time
    end = total_enc_frame * end / total_audio_time
    start = np.ceil(start).astype('int')
    end = np.ceil(end).astype('int')
    return start, end
    