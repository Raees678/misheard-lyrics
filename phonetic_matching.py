from phonemizer import phonemize
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import shutil
import logging
import platform
import re
from typing import Optional, List
import pronouncing
import spacy
_spacy_nlp = spacy.load('en_core_web_sm')
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def word_to_phonemes(word):

    def is_espeak_installed():
        return shutil.which('espeak') is not None or shutil.which('espeak-ng') is not None

    if is_espeak_installed():
        return phonemize(
                [word],
                language='en-us',
                backend='espeak',
                strip=True,
                punctuation_marks=";/,.-!? "
            )[0]
    else:
        print("espeak not installed")

    #if espeak is not available, try to use pronouncing (CMU / ARPAbet)
    if pronouncing is not None:
        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            return phones[0]


    #as a final fallback, return the original word
    return word



    

def words_to_phonemes(words):
    if not words:
        return []

    if shutil.which('espeak') is not None or shutil.which('espeak-ng') is not None:
        return phonemize(
            words,
            language='en-us',
            backend='espeak',
            strip=True,
            punctuation_marks=";/,.-!? "
        )

    #pronouncing as fallback
    if pronouncing is not None:
        res = []
        for w in words:
            phones = pronouncing.phones_for_word(w.lower())
            res.append(phones[0] if phones else '')
        return res

    #as a final fallback, return the original word
    return words


def load_dictionary(limit=None, min_len=2):
    #wordfreq
    try:
        from wordfreq import top_n_list
        words = top_n_list('en', n=limit or 100000)
        return [w for w in words if w.isalpha() and len(w) >= min_len][:limit]
    except Exception:
        pass




def build_noun_list(limit: Optional[int] = None) -> List[str]:
    #get common nouns
    freq_limit = limit 
    common_words = []
    from wordfreq import top_n_list
    common_words = top_n_list('en', n=freq_limit)
    common_words = load_dictionary(limit=freq_limit)


    from nltk.corpus import wordnet as wn
    nouns = []
    for w in common_words:
        if wn.synsets(w, pos=wn.NOUN):
            nouns.append(w)
    return nouns


def precompute_embeddings_for_wordlist(word_list: List[str], batch_size: int = 512):
    model = get_model()
    
    #get syllable counts for all words
    syllable_counts = [syllable_count(w) for w in word_list]
    
    #convert words to phonemes in batches
    n = len(word_list)
    emb_list = []
    for i in range(0, n, batch_size):
        batch = word_list[i:i+batch_size]
        batch_ph = words_to_phonemes(batch)
        final_ph = [p if p else w for p, w in zip(batch_ph, batch)]
        emb = model.encode(final_ph, show_progress_bar=False)
        #normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        emb_list.append(emb)
    if emb_list:
        embeddings = np.vstack(emb_list)
    else:
        embeddings = np.zeros((0, model.encode([''])[0].shape[0]))

    #build a frequency rank map
    rank_map = {w: i for i, w in enumerate(word_list)}
    return word_list, embeddings, rank_map, syllable_counts


def preserve_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def syllable_count(word: str) -> int:
    w = word.lower()
    if pronouncing is not None:
        phones = pronouncing.phones_for_word(w)
        if phones:
            return pronouncing.syllable_count(phones[0])

    #fallback heuristic: count vowel groups (this is bad but better than nothing)
    groups = re.findall(r'[aeiouy]+', w)
    return max(1, len(groups))


def find_best_phonetic_match(target_word: str, word_list: List[str], word_emb: np.ndarray, rank_map: dict, syllable_counts: List[int], top_n: int = 20) -> str:
    ph = word_to_phonemes(target_word)
    model = get_model()
    emb = model.encode([ph], show_progress_bar=False)
    nrm = np.linalg.norm(emb)
    if nrm == 0:
        nrm = 1.0
    emb = emb / nrm
    scores = np.dot(word_emb, emb.T).reshape(-1)
    
    #get top n indices by score
    top_indices = np.argsort(scores)[-top_n:][::-1]
    
    #prefer candidates with same syllable count as target when possible
    target_syllables = syllable_count(target_word)
    same_syllable_indices = [idx for idx in top_indices if syllable_counts[idx] == target_syllables]
    candidates = same_syllable_indices if same_syllable_indices else list(top_indices)

    #return the candidate that is most frequent according to rank_map
    best_idx = min(candidates, key=lambda idx: rank_map.get(word_list[idx], float('inf')))
    return word_list[best_idx]
    


def replace_nouns_and_verbs_in_file(input_path: str, output_path: Optional[str] = None, noun_limit: Optional[int] = 50000, batch_size: int = 512):
    import random

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    
    #build noun candidates and precompute embeddings
    words = build_noun_list(limit=noun_limit)

    word_list, word_emb, rank_map, syllable_counts = precompute_embeddings_for_wordlist(words, batch_size=batch_size)

    #split text into lines
    lines = text.split('\n')
    new_lines = []
    total_replaced = 0
    
    for line in lines:
        #extract words and their positions using regex
        word_pattern = r'\b\w+\b'
        matches = list(re.finditer(word_pattern, line))
        
        #only process lines with words longer than 3 letters
        candidates = [m for m in matches if len(m.group()) > 3]
        
        #randomly select up to 3 words to replace
        to_replace = random.sample(candidates, min(3, len(candidates))) if candidates else []
        
        new_line = line
        replaced_count = 0
        
        #process replacements in reverse order to maintain offsets
        for match in sorted(to_replace, key=lambda m: m.start(), reverse=True):
            word = match.group()

            #find top 10 phonetically similar words and pick the most frequent in english languGE
            replacement = find_best_phonetic_match(word, word_list, word_emb, rank_map, syllable_counts, top_n=20)
            replacement = preserve_case(replacement, word)
            start, end = match.span()
            new_line = new_line[:start] + replacement + new_line[end:]
            replaced_count += 1

        
        total_replaced += replaced_count
        new_lines.append(new_line)
    
    new_text = '\n'.join(new_lines)
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + '.replaced' + (ext or '.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_text)

    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phonetic matching utilities')
    parser.add_argument('--replace', '-r', nargs='?', const='pioneer_lyrics.txt',
                        help='Path to lyrics file to process (default: pioneer_lyrics.txt if present)')
    parser.add_argument('--noun-limit', type=int, default=50000, help='Max noun candidates to consider')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for embedding computation')
    args = parser.parse_args()

    if args.replace:
        input_path = args.replace
        if not os.path.exists(input_path):
            print(f"Replace file not found: {input_path}")
        else:
            print(f"Processing {input_path}, takes around a minute usually")

            out_path = replace_nouns_and_verbs_in_file(input_path, noun_limit=args.noun_limit, batch_size=args.batch_size)
            print(f"Wrote replaced lyrics to: {out_path}")

