
CORPUS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/mfa/common_voice/test/corpus/"

# imports 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "../..")))
from tqdm.auto import tqdm
from datasets import load_dataset, Audio
from utils import DATA_KEY, TEXT_KEY

if not os.path.exists(CORPUS_PATH):
    os.makedirs(CORPUS_PATH)

# load audio dataset
data = load_dataset(DATA_KEY['common_voice'], 'fr', split='test', verification_mode="all_checks")
data = data.cast_column("audio", Audio(sampling_rate=16_000))

# build data format
progress_bar = tqdm(range(len(data)))
for ex in range(len(data)):
    ## Audio
    audio_path = data[ex]['audio']['path']
    # manual address fix
    splits = audio_path.split('/')
    audio_path = "/".join(splits[:-1]) + f"/fr_test_0/" + splits[-1]
    os.system(f"cp {audio_path} {CORPUS_PATH}{ex}.{audio_path.split('.')[-1]}")

    ## Text
    transcript = data[ex][TEXT_KEY['common_voice']]
    # suggested cleaning
    if transcript.startswith('"') and transcript.endswith('"'):
        transcript = transcript[1:-1]
    #
    file = open(f"{CORPUS_PATH}{ex}.txt", 'w')
    file.write(transcript)
    file.close()

    progress_bar.update(1)