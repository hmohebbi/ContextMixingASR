

TASK = "common_voice"
SPLIT = "test" 
SEED = 42
GENERATED_IDS_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/predictions/{TASK}/{SPLIT}/"
ALIGNMENT_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/mfa/common_voice/test/outputs/"
ANNOTATED_DATA_PATH = f"/home/hmohebbi/Projects/ContextMixing_ASR/directory/datasets/{TASK}/{SPLIT}/"

# import
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
import spacy
from inflecteur import inflecteur
import pickle
import torch
import string
from transformers import WhisperProcessor, Wav2Vec2Processor
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
from utils import det_words, irregular_nouns
from utils import MODEL_PATH, DATA_KEY, TEXT_KEY

np.random.seed(SEED)

if not os.path.exists(ANNOTATED_DATA_PATH):
    os.makedirs(ANNOTATED_DATA_PATH)

nlp = spacy.load("fr_core_news_md")
inflecteur = inflecteur()
inflecteur.load_dict()

PROCESSOR = {
    'whisper-base': WhisperProcessor.from_pretrained(MODEL_PATH['whisper-base'], task='transcribe', language='french'),
    'whisper-small': WhisperProcessor.from_pretrained(MODEL_PATH['whisper-small'], task='transcribe', language='french'),
    'whisper-medium': WhisperProcessor.from_pretrained(MODEL_PATH['whisper-medium'], task='transcribe', language='french'),
    'wav2vec2-large-xlsr-53-french': Wav2Vec2Processor.from_pretrained(MODEL_PATH['wav2vec2-large-xlsr-53-french']), 
    'asr-wav2vec2-french': Wav2Vec2Processor.from_pretrained(MODEL_PATH['asr-wav2vec2-french']), 
}

def find_indices(processor, ex, cue_word, target_word):
    # find decoder indices
    target_token_dec_indices = {}
    cue_token_dec_indices = {}
    for model_name in MODEL_PATH.keys():
        if model_name.split('-')[0] == "whisper":
            # mapping subwords to words
            generated_tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[model_name][ex].tolist())
            generated_words = []
            word_indices = []
            current_word = -1
            for token in generated_tokens:
                if token.startswith("Ġ") or token in ['<|fr|>', '<|transcribe|>', '<|notimestamps|>', '<|endoftext|>'] or token in string.punctuation or generated_words[-1] in string.punctuation:
                    generated_words.append(processor.tokenizer.convert_tokens_to_string(token).strip().lower() if token.startswith("Ġ") else token.lower())
                    current_word += 1
                else:
                    generated_words[-1] = generated_words[-1] + processor.tokenizer.convert_tokens_to_string(token).lower()
                word_indices.append(current_word)
            generated_words = np.array(generated_words)
            word_indices = np.array(word_indices)
            
            # find target and cue token indices
            target_word_indices = np.where(generated_words == target_word.lower())[0]
            cue_word_indices = np.where(generated_words == cue_word.lower())[0]
            # Multiple cues:
            # cue word wouldn't come after the target word
            cue_word_indices = cue_word_indices[cue_word_indices < np.max(target_word_indices)]
            # if cue ids are not consecutive that means they are not splited tokens blong to one word. we have multiple same cues so the right cue is the one nearest to the target
            while np.max(cue_word_indices) - np.min(cue_word_indices) > 1:
                cue_word_indices = np.delete(cue_word_indices, np.where(cue_word_indices == np.min(cue_word_indices)))
            # Multiple targets:
            target_word_indices = target_word_indices[target_word_indices > np.min(cue_word_indices)]
            while np.max(target_word_indices) - np.min(target_word_indices) > 1:
                target_word_indices = np.delete(target_word_indices, np.where(target_word_indices == np.max(target_word_indices)))

            # check if there are many
            if len(target_word_indices) > 1:
                print("multiple target words error")
            if len(cue_word_indices) > 1:
                print("multiple cue words error")
            target_token_dec_indices[model_name] = np.where(word_indices == target_word_indices)[0].tolist()
            cue_token_dec_indices[model_name] = np.where(word_indices == cue_word_indices)[0].tolist()
        else:
            # wav2vec based models do not have decoder part
            target_token_dec_indices[model_name] = None
            cue_token_dec_indices[model_name] = None

    # find encoder indices            
    aligned_enc_words = [alignments[ex]['intervals'][i]['word'].lower() for i in range(len(alignments[ex]['intervals']))]
    target_word_enc_indices = np.where(np.isin(np.array(aligned_enc_words), np.array(target_word.lower())))[0]
    cue_word_enc_indices = np.where(np.isin(np.array(aligned_enc_words), np.array(cue_word.lower())))[0]
    # Multiple Cues
    # cue word wouldn't come after the target word
    cue_word_enc_indices = cue_word_enc_indices[cue_word_enc_indices < np.max(target_word_enc_indices)]
    # if cue ids are not consecutive that means they are not splited tokens blong to one word. we have multiple same cues so the right cue is the one nearest to the target
    while np.max(cue_word_enc_indices) - np.min(cue_word_enc_indices) > 1:
        cue_word_enc_indices = np.delete(cue_word_enc_indices, np.where(cue_word_enc_indices == np.min(cue_word_enc_indices)))
    # Multiple Targets
    target_word_enc_indices = target_word_enc_indices[target_word_enc_indices > np.min(cue_word_enc_indices)]
    while np.max(target_word_enc_indices) - np.min(target_word_enc_indices) > 1:
        target_word_enc_indices = np.delete(target_word_enc_indices, np.where(target_word_enc_indices == np.max(target_word_enc_indices)))
    
    cue_word_enc_indices = cue_word_enc_indices.tolist()
    target_word_enc_indices = target_word_enc_indices.tolist()

    return cue_token_dec_indices, target_token_dec_indices, cue_word_enc_indices, target_word_enc_indices
 
def check_ending_letters(verb, tense, person, number):
    cnd = True
    if tense == "Tense_pres":
        if person == "Person_one":
            cnd = cnd and len(verb) > 1 and verb[-1] == 'e'
        elif person == "Person_two":
            cnd = cnd and len(verb) > 2 and verb[-2:] == 'es'
        else: # Person_three
            if number == "Number_plur":
                cnd = cnd and len(verb) > 3 and verb[-3:] == 'ent'
            else: # Number_sing
                cnd = cnd and len(verb) > 1 and verb[-1] == 'e'
    elif tense == "Tense_imp":
        if person == "Person_one":
            cnd = cnd and len(verb) > 3 and verb[-3:] == 'ais'
        elif person == "Person_two":
            cnd = cnd and len(verb) > 3 and verb[-3:] == 'ais'
        else: # Person_three
            if number == "Number_plur":
                cnd = cnd and len(verb) > 5 and verb[-5:] == 'aient'
            else: # Number_sing
                cnd = cnd and len(verb) > 3 and verb[-3:] == 'ait'
    
    if not cnd:
        return False
    
    # check for other forms 
    if person == "Person_three":
        cnd = cnd and (inflecteur.inflect_sentence(verb, tense="Présent", number='s')[-1] == 'e' and
                       inflecteur.inflect_sentence(verb, tense="Présent", number='p')[-3:] == 'ent' and
                       inflecteur.inflect_sentence(verb, tense="Imparfait", number='s')[-3:] == 'ait' and
                       inflecteur.inflect_sentence(verb, tense="Imparfait", number='p')[-5:] == 'aient'
                       )
    return cnd

def det_noun(T):
    data = []
    for ex in range(len(T)):
        # filter non successfull forced aligned examples
        if alignments[ex]['intervals'] is None:
            # print("#1")
            continue

        sentence = T[ex]
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]

        doc = nlp(sentence)
        word_list = [word.text for word in doc]
        dep_list = [word.dep_ for word in doc] 

        cue_target_pairs = []
        for ind in range(len(word_list)):
            if dep_list[ind] == "det" and word_list[ind] in det_words:
                # filter irregular nouns
                if doc[ind].head.text in irregular_nouns:
                    # print("#2")
                    continue 
                # filter those nouns which their singualr forms end with al or ail
                if doc[ind].head.lemma_[-2:] == 'al' or doc[ind].head.lemma_[-3:] == 'ail': 
                    # print("#3")
                    continue 
                
                cue_word = doc[ind]
                target_word = doc[ind].head

                # filter if target or cue words have not been founded by aligner
                aligned_words = [alignments[ex]['intervals'][i]['word'].lower() for i in range(len(alignments[ex]['intervals']))]
                if cue_word.text.lower() not in aligned_words or target_word.text.lower() not in aligned_words:
                    # print("#4")
                    continue
                # cue word must come before target word
                if aligned_words.index(cue_word.text.lower()) > aligned_words.index(target_word.text.lower()):
                    continue
                # filter those if target and cue words are not correctly generated by all models under the probe
                not_generated = False
                for model_name in MODEL_PATH.keys():
                    processor = PROCESSOR[model_name]
                    gen_words = processor.tokenizer.decode(generated_ids[model_name][ex].tolist(), skip_special_tokens=True).split()
                    gen_words = [w.lower() for w in gen_words]
                    if cue_word.text.lower() not in gen_words or target_word.text.lower() not in gen_words:
                        not_generated = True
                    else: # cue word must come before target word
                        if gen_words.index(cue_word.text.lower()) > gen_words.index(target_word.text.lower()):
                            not_generated = True
                if not_generated:    
                    # print("#5")
                    continue

                cue_target_pairs.append((cue_word, target_word))
        
        cue_target_pairs = list(set(cue_target_pairs)) # it's difficult to track repeated pairs of cue target in a sentence
        if len(cue_target_pairs) == 0:
            continue

        # find indices
        for cue_word, target_word in cue_target_pairs:
            cue_token_dec_indices, target_token_dec_indices, cue_word_enc_indices, target_word_enc_indices = find_indices(PROCESSOR['whisper-small'], ex, cue_word.text, target_word.text)

            labels = {'number': target_word.morph.get('number'),
                    'person': target_word.morph.get('person'),
                    'tense': target_word.morph.get('tense')}
            
            data.append({
                'template': 'det_noun',
                'org_id': ex, 
                'text': sentence, 
                'cue_word': cue_word.text, 
                'target_word': target_word.text,
                'target_word_2': None,
                'path': org_data[ex]['path'],
                'audio': org_data[ex]['audio'],
                'alignment': alignments[ex],
                'target_indices': {'enc': target_word_enc_indices, 'dec': target_token_dec_indices},
                'cue_indices': {'enc': cue_word_enc_indices, 'dec': cue_token_dec_indices},
                'target_indices_2': None,
                'label_number': labels['number'],
                'label_person': labels['person'],
                'label_tense': labels['tense'],
                        })
    return data


def pronoun_verb(T):
    data = []
    for ex in range(len(T)):
        # filter non successfull forced aligned examples
        if alignments[ex]['intervals'] is None:
            # print("#1")
            continue

        sentence = T[ex]
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]

        doc = nlp(sentence)
        word_list = [word.text for word in doc]
        dep_list = [word.dep_ for word in doc] 
        pos_list = [word.pos_ for word in doc]

        if not('nsubj' in dep_list and 'PRON' in pos_list and 'VERB' in pos_list):
            continue 

        cue_word = None
        target_word = None
        for ind in range(len(word_list)):
            # filter examples which doens't include a pronoun and verb 
            if not(pos_list[ind] == "PRON" and dep_list[ind] == "nsubj"):
                continue
            if not(doc[ind].head.pos_ == 'VERB'): 
                continue
            # we're looking for present and impefective tenses only
            if not(doc[ind].head.morph.get('tense') == "Tense_pres" or doc[ind].head.morph.get('tense') == "Tense_imp"):
                continue
            # filter 1pl and 2pl
            if doc[ind].head.morph.get('number') == "Number_plur" and (doc[ind].head.morph.get('person') == "Person_one" or doc[ind].head.morph.get('person') == "Person_two"):
                continue
            
            cue_word = doc[ind]
            target_word = doc[ind].head
            break
        
        if cue_word is None or target_word is None:
            continue
        # filter verbs that are incompatible with inflecteur
        if target_word.text in ['héberge', 'arrose', 'admire', 'adapte', 'efface', 'engouffre', 'asperge', 'aveuglait']:
            continue

        # check ending letters
        if not check_ending_letters(verb=target_word.text, tense=target_word.morph.get('tense'), person=target_word.morph.get('person'), number=target_word.morph.get('number')):
            continue


        # filter if target or cue words have not been founded by aligner
        aligned_words = [alignments[ex]['intervals'][i]['word'].lower() for i in range(len(alignments[ex]['intervals']))]
        if cue_word.text.lower() not in aligned_words or target_word.text.lower() not in aligned_words:
            # print("#4")
            continue
        # cue word must come before target word
        if aligned_words.index(cue_word.text.lower()) > aligned_words.index(target_word.text.lower()):
            continue

        # filter those if target and cue words are not correctly generated by all models under the probe
        not_generated = False
        for model_name in MODEL_PATH.keys():
            processor = PROCESSOR[model_name]
            gen_words = processor.tokenizer.decode(generated_ids[model_name][ex].tolist(), skip_special_tokens=True).split()
            gen_words = [w.lower() for w in gen_words]
            if cue_word.text.lower() not in gen_words or target_word.text.lower() not in gen_words:
                not_generated = True
            else: # cue word must come before target word
                if gen_words.index(cue_word.text.lower()) > gen_words.index(target_word.text.lower()):
                    not_generated = True
        if not_generated:    
            # print("#5")
            continue
        
        # find indices
        cue_token_dec_indices, target_token_dec_indices, cue_word_enc_indices, target_word_enc_indices = find_indices(PROCESSOR['whisper-small'], ex, cue_word.text, target_word.text)
        
        labels = {'number': target_word.morph.get('number'),
                  'person': target_word.morph.get('person'),
                  'tense': target_word.morph.get('tense')}
        
        data.append({
            'template': 'pronoun_verb',
            'org_id': ex, 
            'text': sentence, 
            'cue_word': cue_word.text, 
            'target_word': target_word.text,
            'target_word_2': None,
            'path': org_data[ex]['path'],
            'audio': org_data[ex]['audio'],
            'alignment': alignments[ex],
            'target_indices': {'enc': target_word_enc_indices, 'dec': target_token_dec_indices},
            'cue_indices': {'enc': cue_word_enc_indices, 'dec': cue_token_dec_indices},
            'target_indices_2': None,
            'label_number': labels['number'],
            'label_person': labels['person'],
            'label_tense': labels['tense'],
                    })
        
    return data


def det_noun_verb(T):
    data = []
    for ex in range(len(T)):
        # filter non successfull forced aligned examples
        if alignments[ex]['intervals'] is None:
            continue

        sentence = T[ex]
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]

        doc = nlp(sentence)
        word_list = [word.text for word in doc]
        dep_list = [word.dep_ for word in doc] 
        pos_list = [word.pos_ for word in doc]

        if not('nsubj' in dep_list and 'DET' in pos_list and 'NOUN' in pos_list and 'VERB' in pos_list):
            continue 

        cue_word = None
        target_word = None
        target_word_2 = None
        for ind in range(len(word_list)):
            # find det
            if not(dep_list[ind] == "det" and word_list[ind] in det_words):
                continue

            # find noun
            if not(doc[ind].head.pos_ == "NOUN" and doc[ind].head.dep_ == "nsubj"):
                continue
            # filter irregular nouns
            if doc[ind].head.text in irregular_nouns:
                continue 
            # filter those nouns which their singualr forms end with al or ail
            if doc[ind].head.lemma_[-2:] == 'al' or doc[ind].head.lemma_[-3:] == 'ail': 
                continue 
            
            # find verb
            if not(doc[ind].head.head.pos_ == "VERB"):
                continue
            # we're looking for present and impefective tenses only
            if not(doc[ind].head.head.morph.get('tense') == "Tense_pres" or doc[ind].head.head.morph.get('tense') == "Tense_imp"):
                continue
            # filter 1pl and 2pl
            if doc[ind].head.head.morph.get('number') == "Number_plur" and (doc[ind].head.head.morph.get('person') == "Person_one" or doc[ind].head.head.morph.get('person') == "Person_two"):
                continue
            
            cue_word = doc[ind]
            target_word = doc[ind].head
            target_word_2 = doc[ind].head.head
            break
        
        if cue_word is None or target_word is None or target_word_2 is None:
            continue
        # filter verbs that are incompatible with inflecteur
        if target_word_2.text in ['héberge', 'arrose', 'admire', 'adapte', 'efface', 'engouffre', 'asperge', 'aveuglait']:
            continue

        # check ending letters
        if not check_ending_letters(verb=target_word_2.text, tense=target_word_2.morph.get('tense'), person=target_word_2.morph.get('person'), number=target_word_2.morph.get('number')):
            continue


        # filter if target or cue words have not been founded by aligner
        aligned_words = [alignments[ex]['intervals'][i]['word'].lower() for i in range(len(alignments[ex]['intervals']))]
        if cue_word.text.lower() not in aligned_words or target_word.text.lower() not in aligned_words or target_word_2.text.lower() not in aligned_words:
            continue
        # cue word must come before target word
        if aligned_words.index(cue_word.text.lower()) > aligned_words.index(target_word.text.lower()):
            continue

        # filter those if target and cue words are not correctly generated by all models under the probe
        not_generated = False
        for model_name in MODEL_PATH.keys():
            processor = PROCESSOR[model_name]
            gen_words = processor.tokenizer.decode(generated_ids[model_name][ex].tolist(), skip_special_tokens=True).split()
            gen_words = [w.lower() for w in gen_words]
            if cue_word.text.lower() not in gen_words or target_word.text.lower() not in gen_words or target_word_2.text.lower() not in gen_words:
                not_generated = True
            else: # cue word must come before target word, target word (noun) before target_2 (verb) and cue word (det) before target_2 (verb)
                if gen_words.index(cue_word.text.lower()) > gen_words.index(target_word.text.lower()) or gen_words.index(target_word.text.lower()) > gen_words.index(target_word_2.text.lower()) or gen_words.index(cue_word.text.lower()) > gen_words.index(target_word_2.text.lower()):
                    not_generated = True
        if not_generated:    
            continue
        
        # find indices
        cue_token_dec_indices, target_token_dec_indices, cue_word_enc_indices, target_word_enc_indices = find_indices(PROCESSOR['whisper-small'], ex, cue_word.text, target_word.text)
        cue_token_2_dec_indices, target_token_2_dec_indices, cue_word_2_enc_indices, target_word_2_enc_indices = find_indices(PROCESSOR['whisper-small'], ex, cue_word.text, target_word_2.text)
        if cue_token_dec_indices != cue_token_2_dec_indices or cue_word_enc_indices != cue_word_2_enc_indices:
            continue
        
        labels = {'number': target_word_2.morph.get('number'),
                  'person': target_word_2.morph.get('person'),
                  'tense': target_word_2.morph.get('tense')}
        
        data.append({
            'template': 'det_noun_verb',
            'org_id': ex, 
            'text': sentence, 
            'cue_word': cue_word.text, 
            'target_word': target_word.text,
            'target_word_2': target_word_2.text,
            'path': org_data[ex]['path'],
            'audio': org_data[ex]['audio'],
            'alignment': alignments[ex],
            'cue_indices': {'enc': cue_word_enc_indices, 'dec': cue_token_dec_indices},
            'target_indices': {'enc': target_word_enc_indices, 'dec': target_token_dec_indices},
            'target_indices_2': {'enc': target_word_2_enc_indices, 'dec': target_token_2_dec_indices},
            'label_number': labels['number'],
            'label_person': labels['person'],
            'label_tense': labels['tense'],
                    })
    return data


# load original data
org_data = load_dataset(DATA_KEY[TASK], 'fr', split=SPLIT, verification_mode="all_checks")
org_data = org_data.cast_column("audio", Audio(sampling_rate=16_000))

# Load generated ids
generated_ids = {}
for model_name in MODEL_PATH.keys():
    with open(f'{GENERATED_IDS_PATH}{model_name}/generated_ids.pkl', 'rb') as fp:
        generated_ids[model_name] = pickle.load(fp)

# load alignments
file_ids = [int(f.split('.')[0]) for f in os.listdir(ALIGNMENT_PATH) if f.endswith('.TextGrid')]
alignments = []
for ex in range(len(org_data)):
    if ex not in file_ids:
        alignments.append({'id': ex, 'total_start': None, 'total_end': None, 'intervals': None})           
        continue
    lines = open(f"{ALIGNMENT_PATH}{ex}.TextGrid", "r").readlines()
    total_min = float(lines[3].strip().split()[2])
    total_max = float(lines[4].strip().split()[2])
    num_intervals = int(lines[13].strip().split('=')[-1])
    intervals = []
    for it in range(num_intervals):
        xmin = float(lines[15+it*4].split("=")[-1].strip())
        xmax = float(lines[16+it*4].split("=")[-1].strip())
        text = lines[17+it*4].split("=")[-1].strip()[1:-1]
        if text != "":
            intervals.append({'start': xmin, 'end': xmax, 'word': text})
    alignments.append({'total_start': total_min, 'total_end': total_max, 'intervals': intervals})           
alignments = Dataset.from_list(alignments)


# create and save annotated datasets
# pronoun_verb
pronoun_verb_data = pronoun_verb(org_data[TEXT_KEY[TASK]])
pronoun_verb_data = Dataset.from_list(pronoun_verb_data)
print(len(pronoun_verb_data))

# det_noun_verb
det_noun_verb_data = det_noun_verb(org_data[TEXT_KEY[TASK]])
det_noun_verb_data = Dataset.from_list(det_noun_verb_data)
print(len(det_noun_verb_data))

# det_noun
det_noun_data = det_noun(org_data[TEXT_KEY[TASK]])
det_noun_data = Dataset.from_list(det_noun_data)
# balancing sg and pl in det_noun template
sg_indices = np.where(np.array(det_noun_data['label_number']) == 'Number_sing')[0]
pl_indices = np.where(np.array(det_noun_data['label_number']) == 'Number_plur')[0]
np.random.shuffle(sg_indices)
np.random.shuffle(pl_indices)
n = (1000 - (len(pronoun_verb_data)+len(det_noun_verb_data))) // 2
balanced_det_noun_data = concatenate_datasets([det_noun_data.select(sg_indices[:n]), det_noun_data.select(pl_indices[:n])])
print(len(balanced_det_noun_data))

# aggregate all templates
all_data = concatenate_datasets([balanced_det_noun_data, pronoun_verb_data, det_noun_verb_data])
all_data.save_to_disk(f"{ANNOTATED_DATA_PATH}all")


