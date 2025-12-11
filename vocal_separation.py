import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import subprocess
import time
import sys

# DEVICE SETUP


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


# VOCAL SEPARATION WITH HTDEMUCS


def load_demucs_model(device="cpu"):
    from demucs.pretrained import get_model

    model = get_model(name="htdemucs", repo=None)
    model.to(device)
    model.eval()

    print(f"Loaded htdemucs model")
    print(f"Available stems: {model.sources}")
    print(f"Sample rate: {model.samplerate}")

    return model


def load_audio_track(audio_path, model):
    from demucs.separate import load_track

    sr = model.samplerate
    # Load as stereo (2 channels) because demucs expects stereo input to separate into stems
    # We dont use this function when we load the audio for transcription
    audio = load_track(audio_path, 2, sr)

    print(f"Loaded audio: {audio_path}")
    print(f"Shape: {audio.shape}")
    print(f"Duration: {audio.shape[1] / sr:.2f} seconds")

    return audio, sr


def normalize_audio(audio):
    # Normalize audio using mean and standard deviation
    # This doesnt change the number of channels, it
    ref = audio.mean(0)
    audio_normalized = (audio - ref.mean()) / ref.std()
    return audio_normalized


def separate_stems(model, audio_normalized, device="cpu"):
    # Separate audio into individual stems (drums, bass, other, vocals)
    from demucs.apply import apply_model

    with torch.no_grad():
        sources = apply_model(
            model,
            audio_normalized[
                None
            ],  # Adds batch dimension to the audio because the model expects a batch dimension
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )

    return sources


def extract_vocals(sources, model):
    # Get the index of vocals in the sources
    vocal_idx = model.sources.index("vocals")

    # sources shape: [1, 4, 2, num_samples]
    # Extract vocals: [2, num_samples] -> take first channel for mono
    vocals = sources[0, vocal_idx]

    return vocals


def extract_instrumentals(sources, model):
    vocal_idx = model.sources.index("vocals")

    # Sum all non-vocal stems
    instrumentals = None
    for i, source_name in enumerate(model.sources):
        if i != vocal_idx:
            if instrumentals is None:
                instrumentals = sources[0, i]
            else:
                instrumentals = instrumentals + sources[0, i]

    return instrumentals


def separate_vocals_from_audio(audio_path, output_dir=None, device=None):
    # Takes an audio path, loads the demucs model
    # loads the audio in stereo, normalizes it,
    # separates the stems and then saves the vocals and instrumentals to the output directory
    if device is None:
        device = get_device()

    # Load model
    model = load_demucs_model(device)

    # Load audio
    audio, sr = load_audio_track(audio_path, model)

    # Normalize
    audio_normalized = normalize_audio(audio)

    # Separate stems
    print("Separating stems...")
    sources = separate_stems(model, audio_normalized, device)

    # Extract vocals and instrumentals
    vocals = extract_vocals(sources, model)
    instrumentals = extract_instrumentals(sources, model)

    result = {
        "vocals": vocals,
        "instrumentals": instrumentals,
        "sample_rate": sr,
        "sources": sources,
        "model": model,
    }

    # Save outputs if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_name = Path(audio_path).stem

        # Save vocals (mono)
        vocals_mono = vocals[0].cpu().numpy()
        vocals_path = output_dir / f"{audio_name}_vocals.wav"
        sf.write(str(vocals_path), vocals_mono, sr)
        print(f"Saved vocals to: {vocals_path}")

        # Save instrumentals (mono)
        instrumentals_mono = instrumentals[0].cpu().numpy()
        instrumentals_path = output_dir / f"{audio_name}_instrumental.wav"
        sf.write(str(instrumentals_path), instrumentals_mono, sr)
        print(f"Saved instrumentals to: {instrumentals_path}")

        result["vocals_path"] = vocals_path
        result["instrumentals_path"] = instrumentals_path

    return result


# WAV2VEC2 SPEECH RECOGNITION


def load_wav2vec2_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model_names = {
        "english": "facebook/wav2vec2-large-960h-lv60-self",
    }

    model_name = model_names["english"]

    print(f"Loading Wav2Vec2 model: {model_name}")
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    print(f"Vocabulary size: {len(processor.tokenizer.get_vocab())}")

    return model, processor


def resample_audio(audio, orig_sr, target_sr=16000):
    # Resample audio to target sample rate
    if orig_sr == target_sr:
        return audio

    # Convert to numpy if tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # If stereo, convert to mono
    if audio.ndim == 2:
        audio = audio[0]  # Take first channel

    # Resample
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return resampled


# PARAKEET (NEMO) SPEECH RECOGNITION


def load_parakeet_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    # Returns the Parakeet model
    import nemo.collections.asr as nemo_asr

    print(f"Loading Parakeet model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    return model


def transcribe_with_parakeet(audio_path, model=None):
    # Gets words with timestamps from Parakeet
    if model is None:
        model = load_parakeet_model()

    print(f"Transcribing with Parakeet: {audio_path}")

    # Transcribe with timestamps
    output = model.transcribe(
        [str(audio_path)],
        timestamps=True,
        return_hypotheses=True,
    )

    # Extract transcription and word timestamps
    hypothesis = output[0]
    transcription = hypothesis.text

    # Convert Parakeet timestamp format to our format: (word, start, end)
    word_timestamps = hypothesis.timestamp.get("word", [])
    words_with_timestamps = [
        (ts["word"], ts["start"], ts["end"]) for ts in word_timestamps
    ]

    print(f"Transcribed {len(words_with_timestamps)} words")

    return {
        "transcription": transcription,
        "words_with_timestamps": words_with_timestamps,
        "model": model,
        "raw_output": output,
    }


# NOISE INJECTION


def add_noise_to_audio(audio_path, noise_amplitude=0.07, output_path=None):
    # Adds whit noise to the audio using ffmpeg
    audio_path = Path(audio_path)

    if output_path is None:
        output_path = (
            audio_path.parent
            / f"{audio_path.stem}_noisy_{noise_amplitude}{audio_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    if output_path.exists():
        print(f"Noisy audio already exists: {output_path}")
        return str(output_path)

    print(f"Adding white noise (amplitude={noise_amplitude}) to: {audio_path}")

    # Get sample rate from input file
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    sample_rate = result.stdout.strip() or "16000"

    # Add noise using ffmpeg
    filter_complex = (
        f"[0:a]volume=1[main];"
        f"anoisesrc=c=white:r={sample_rate}:a={noise_amplitude}[noise];"
        f"[main][noise]amix=inputs=2:duration=first"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_complex,
            str(output_path),
        ],
        capture_output=True,
    )

    print(f"Created noisy audio: {output_path}")
    return str(output_path)


# WAV2VEC2 STUFF


def get_audio_segments(audio, window_seconds=15, sample_rate=16000):
    # Use librosa to chop the audio into segments of 15 seconds
    window_size = int(sample_rate * window_seconds)
    hop_length = window_size

    # Pad last incomplete frame with 0 if the audio is not a multiple of the window size
    pad_length = (window_size - (len(audio) % window_size)) % window_size
    padded_audio = np.pad(audio, (0, pad_length), mode="constant")

    segments = librosa.util.frame(
        padded_audio, frame_length=window_size, hop_length=hop_length, axis=0
    )

    return segments


def get_logits_from_segment(segment, model, processor, sample_rate=16000):
    # Process a single segment with the Wav2Vec2 model and return the logits
    # This processor is a wav2vec2 processor, it takes the segment and does what the model wants to it
    input_vals = processor(
        segment, return_tensors="pt", padding="longest", sampling_rate=sample_rate
    ).input_values

    with torch.no_grad():
        seg_logits = model(input_vals).logits

    return seg_logits


def get_all_logits(audio, model, processor, sample_rate=16000, window_seconds=15):
    segments = get_audio_segments(audio, window_seconds, sample_rate)
    print(f"Processing {len(segments)} segments...")

    logits = None
    for i, seg in enumerate(segments):
        seg_logits = get_logits_from_segment(seg, model, processor, sample_rate)

        if logits is None:
            logits = seg_logits
        else:
            logits = torch.cat((logits, seg_logits), dim=1)

        print(f"  Processed segment {i + 1}/{len(segments)}")

    return logits


def get_emission(logits):
    # Apply log softmax to the logits to get emissions
    emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()
    return emission


def get_transcription(logits, processor):
    # Pick the best emission probability for each time step greedily
    pred = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred)
    return transcription[0]


def download_language_model(model_dir="./lm_models"):
    # Downloads the 4-gram language model from the OpenSLR archive
    # Used with Wav2Vec2 for English ASR
    import urllib.request
    import gzip
    import shutil
    from pathlib import Path

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # LibriSpeech 4-gram LM in ARPA format (designed for ASR, ~1.4GB compressed)
    lm_url = "https://www.openslr.org/resources/11/4-gram.arpa.gz"
    lm_path_gz = model_dir / "4-gram.arpa.gz"
    lm_path = model_dir / "4-gram.arpa"

    if lm_path.exists():
        print(f"Language model already exists at: {lm_path}")
        return str(lm_path)

    # Download compressed file
    if not lm_path_gz.exists():
        print(f"Downloading language model to: {lm_path_gz}")
        print("This may take several minutes (~1.4GB)...")
        urllib.request.urlretrieve(lm_url, str(lm_path_gz))
        print("Download complete.")

    # Decompress
    print(f"Decompressing to: {lm_path}")
    with gzip.open(str(lm_path_gz), "rb") as f_in:
        with open(str(lm_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Clean up compressed file
    lm_path_gz.unlink()

    print(f"Language model ready at: {lm_path}")
    return str(lm_path)


def get_unigrams_from_arpa(arpa_path, max_unigrams=150000):
    # Extracts maximum 150000 unigrams from the ARPA file
    unigrams = []
    in_unigrams = False

    print(f"Extracting unigrams from: {arpa_path}")

    with open(arpa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Keeps going until it finds the unigrams section
            if line == "\\1-grams:":
                in_unigrams = True
                continue
            elif line.startswith("\\") and in_unigrams:
                # End of unigrams section
                break
            # If you are in the unigrams section and the line is not empty, you split the line into parts and add the word to the unigrams list
            if in_unigrams and line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    # The word is the second part of the line
                    word = parts[1]
                    # Skip special tokens
                    if word not in ["<s>", "</s>", "<unk>"]:
                        unigrams.append(word)
                        if len(unigrams) >= max_unigrams:
                            break

    print(f"Extracted {len(unigrams)} unigrams")
    return unigrams


def build_ctc_decoder(processor, lm_path=None, alpha=0.5, beta=1.0):
    from pyctcdecode import build_ctcdecoder

    # Get vocabulary from processor. This comes from the Wav2Vec2 model.
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Build labels list for pyctcdecode
    # - Index 0 (<pad>) is the CTC blank token, must be ""
    # - | is the word boundary, should become " "
    # - Other special tokens need unique placeholders to avoid duplicates
    labels = []
    for i, (label, idx) in enumerate(sorted_vocab):
        if label == "<pad>":
            # CTC blank token outputted at index 0, make it empty string
            labels.append("")
        elif label == "|":
            # Word boundary becomes space
            labels.append(" ")
        elif label in ["<s>", "</s>", "<unk>"]:
            # Use unique placeholder for other special tokens
            # Convert them into <s> or </s> or <unk>
            # Without this they go into the else, become "" and raise an error.
            # These won't match in LM but avoids duplicate "" error
            labels.append(f"⟨{label[1:-1]}⟩")
        else:
            # Uppercase letters for Wav2Vec2 vocab
            labels.append(label.upper())

    if lm_path:
        print(f"Building decoder with language model: {lm_path}")

        # Extract unigrams from ARPA file for better decoding
        unigrams = None
        if lm_path.endswith(".arpa"):
            unigrams = get_unigrams_from_arpa(lm_path)

        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=lm_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )
    else:
        print("Building decoder without language model (beam search only)")
        decoder = build_ctcdecoder(labels)

    return decoder


def get_transcription_with_lm(logits, decoder, beam_width=100):
    # Use the languahe model with beam search and the unigrams to get the
    # Convert to numpy and get probabilities
    logits_np = logits[0].cpu().numpy()

    # Decode with beam search
    transcription = decoder.decode(logits_np, beam_width=beam_width)

    return transcription


def get_transcription_with_timestamps(
    logits, decoder, sample_rate=16000, beam_width=100
):
    """
    Get transcription with word-level timestamps.

    Args:
        logits: Model output logits
        decoder: pyctcdecode decoder
        sample_rate: Audio sample rate
        beam_width: Beam width for search

    Returns:
        tuple: (transcription_text, list of (word, start_time, end_time) tuples)
    """
    logits_np = logits[0].cpu().numpy()

    # Decode with beam search - returns list of (text, frames, logit_score, lm_score)
    beams = decoder.decode_beams(logits_np, beam_width=beam_width)

    if not beams:
        return "", []

    # Best beam result
    best_beam = beams[0]
    text = best_beam[0]  # transcription text
    word_frames = best_beam[2]  # list of (word, (start_frame, end_frame))

    # The frame rate for Wav2Vec2-large is sample_rate / 320
    # (320 is the stride of the convolutional feature extractor)
    frame_rate = sample_rate / 320

    words_with_timestamps = []
    for word, (start_frame, end_frame) in word_frames:
        start_time = start_frame / frame_rate
        end_time = end_frame / frame_rate
        words_with_timestamps.append((word, start_time, end_time))

    return text, words_with_timestamps


def format_timestamp(seconds):
    # Formats the timestamp into a readable format
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def print_timestamped_transcription(words_with_timestamps):
    print("\n" + "=" * 60)
    print("TRANSCRIPTION WITH TIMESTAMPS:")
    print("=" * 60)

    for word, start, end in words_with_timestamps:
        print(f"[{format_timestamp(start)} -> {format_timestamp(end)}] {word}")

    print("=" * 60)


# region INTERACTIVE KARAOKE-STYLE PLAYBACK

# ANSI escape sequences for terminal styling
HIGHLIGHT = "\x1b[7m"  # Reverse/inverted colors
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
RED_BG = "\x1b[41m"  # Red background for highlighted wrong word


# LYRICS COMPARISON


def normalize_word_for_comparison(word):
    # Normalize a word for comparison by removing punctuation.
    # Keeps apostrophes for contractions.
    import re

    # Remove all punctuation except apostrophes, then lowercase
    normalized = re.sub(r"[^\w']", "", word).lower()
    return normalized


def load_reference_lyrics(lyrics_path):
    # Load reference lyrics from a text file.
    # Returns a tuple of (original_words, normalized_words)
    # - original_words: List of words preserving punctuation for display
    # - normalized_words: List of lowercased words with punctuation removed for comparison
    import re

    with open(lyrics_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into words, preserving punctuation attached to words
    original_words = text.split()

    # Create normalized versions for comparison
    normalized_words = [normalize_word_for_comparison(w) for w in original_words]

    return original_words, normalized_words


def align_words(transcribed_words, reference_words, reference_words_original=None):
    # Align transcribed words with reference words using dynamic programming, edit distance
    # If no original words provided, use the reference words as-is
    if reference_words_original is None:
        reference_words_original = reference_words

    # Normalize transcribed words for comparison (strip punctuation)
    trans = [normalize_word_for_comparison(w) for w in transcribed_words]
    # Reference words should already be normalized, but ensure lowercase
    ref = [w.lower() for w in reference_words]

    m, n = len(trans), len(ref)

    # DP table for edit distance with backtracking
    # dp[i][j] = min edits to align trans[:i] with ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i  # All insertions
    for j in range(n + 1):
        dp[0][j] = j  # All deletions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if trans[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Insertion (extra word in transcription)
                    dp[i][j - 1],  # Deletion (missing word in transcription)
                    dp[i - 1][j - 1],  # Substitution (wrong word)
                )

    # Backtrack to find alignment
    alignment = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trans[i - 1] == ref[j - 1]:
            # Use original reference word (with punctuation) for display
            alignment.append(
                (transcribed_words[i - 1], "correct", reference_words_original[j - 1])
            )
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution - use original reference word for display
            alignment.append(
                (transcribed_words[i - 1], "wrong", reference_words_original[j - 1])
            )
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Insertion (extra word in transcription)
            alignment.append((transcribed_words[i - 1], "inserted", None))
            i -= 1
        elif j > 0:
            # Deletion (missing word) - use original reference word for display
            alignment.append(
                (
                    reference_words_original[j - 1],
                    "missing",
                    reference_words_original[j - 1],
                )
            )
            j -= 1

    alignment.reverse()

    # Calculate statistics
    correct = sum(1 for _, status, _ in alignment if status == "correct")
    wrong = sum(1 for _, status, _ in alignment if status == "wrong")
    inserted = sum(1 for _, status, _ in alignment if status == "inserted")
    missing = sum(1 for _, status, _ in alignment if status == "missing")

    stats = {
        "correct": correct,
        "wrong": wrong,
        "inserted": inserted,
        "missing": missing,
        "total_ref": len(reference_words_original),
        "total_trans": len(transcribed_words),
        "edit_distance": dp[m][n],
        "accuracy": (
            (correct / len(reference_words_original) * 100)
            if reference_words_original
            else 0
        ),
    }

    return alignment, stats


def colorize_word(word, status, is_current=False):
    # Apply color to a word based on its alignment status.
    if status == "correct":
        if is_current:
            return f"{HIGHLIGHT}{GREEN} {word} {RESET}"
        return f"{GREEN}{word}{RESET}"
    elif status == "wrong":
        if is_current:
            return f"{RED_BG} {word} {RESET}"
        return f"{RED}{word}{RESET}"
    elif status == "inserted":
        if is_current:
            return f"{RED_BG} {word} {RESET}"
        return f"{RED}[{word}]{RESET}"  # Brackets indicate inserted
    elif status == "missing":
        return f"{RED}•{word}•{RESET}"  # Show missing word between dots
    else:
        if is_current:
            return f"{HIGHLIGHT} {word} {RESET}"
        return word


def print_comparison_stats(stats):
    print("\n" + "=" * 60)
    print("  LYRICS COMPARISON")
    print("=" * 60)
    print(f"  Reference words:    {stats['total_ref']}")
    print(f"  Transcribed words:  {stats['total_trans']}")
    print(f"  {GREEN}Correct:{RESET}            {stats['correct']}")
    print(f"  {RED}Wrong:{RESET}              {stats['wrong']}")
    print(f"  {RED}Inserted:{RESET}           {stats['inserted']}")
    print(f"  {RED}Missing:{RESET}            {stats['missing']}")
    print(f"  Edit distance:      {stats['edit_distance']}")
    print(f"  Accuracy:           {stats['accuracy']:.1f}%")
    print("=" * 60)


def print_colored_transcription(alignment):
    # Print the full transcription with colors.
    print("\n" + "=" * 60)
    print("  COLORED TRANSCRIPTION")
    print(
        f"  {GREEN}Green{RESET} = correct, {RED}Red{RESET} = wrong/inserted, {RED}•{RESET} = missing"
    )
    print("=" * 60 + "\n")

    line = []
    line_len = 0

    for word, status, _ in alignment:
        colored = colorize_word(word, status)
        word_len = len(word) + 1

        if line_len + word_len > 70:
            print("  " + " ".join(line))
            line = []
            line_len = 0

        line.append(colored)
        line_len += word_len

    if line:
        print("  " + " ".join(line))

    print("\n" + "=" * 60)


def get_current_words(
    words_with_timestamps, current_time, lookbehind=3, lookahead=6, alignment=None
):
    """
    Get a window of words around the current playback time.

    The current word is highlighted and stays at a fixed position for stability.
    If alignment is provided, words are colored based on correctness.

    Args:
        words_with_timestamps: List of (word, start_time, end_time) tuples
        current_time: Current playback time in seconds
        lookbehind: Number of words to show before current word
        lookahead: Number of words to show after current word
        alignment: Optional alignment list from align_words() for color coding

    Returns:
        Formatted string with current word highlighted
    """
    if not words_with_timestamps:
        return ""

    current_idx = 0

    # Find current word index based on time
    for i, (word, start, end) in enumerate(words_with_timestamps):
        if start <= current_time <= end:
            current_idx = i
            break
        elif start > current_time:
            current_idx = max(0, i - 1)
            break
        current_idx = i

    # Build output: lookbehind words + current word (highlighted) + lookahead words
    words = []
    start_idx = max(0, current_idx - lookbehind)
    end_idx = min(len(words_with_timestamps), current_idx + lookahead + 1)

    # Pad with empty spaces if we're near the beginning (keeps position stable)
    padding_needed = lookbehind - (current_idx - start_idx)
    for _ in range(padding_needed):
        words.append("      ")

    # Build mapping from transcribed word index to alignment index
    # (alignment may have 'missing' entries that don't correspond to transcribed words)
    trans_to_align = {}
    if alignment:
        trans_idx = 0
        for align_idx, (_, status, _) in enumerate(alignment):
            if status != "missing":
                trans_to_align[trans_idx] = align_idx
                trans_idx += 1

    for i in range(start_idx, end_idx):
        word = words_with_timestamps[i][0]
        is_current = i == current_idx

        if alignment and i in trans_to_align:
            _, status, _ = alignment[trans_to_align[i]]
            colored = colorize_word(word, status, is_current)
            if i < current_idx and status == "correct":
                # Dim past correct words
                colored = f"{DIM}{GREEN}{word}{RESET}"
            words.append(colored)
        else:
            # No alignment info, use default styling
            if is_current:
                words.append(f"{HIGHLIGHT} {word} {RESET}")
            elif i < current_idx:
                words.append(f"{DIM}{word}{RESET}")
            else:
                words.append(word)

    return " ".join(words)


def get_audio_duration(audio_path):
    """Get audio duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def play_audio_with_lyrics(
    audio_path, words_with_timestamps, alignment=None, stats=None
):
    # Play audio and display synchronized lyrics karaoke-style.

    duration = get_audio_duration(audio_path)

    print("\n" + "=" * 70)
    print(f"  {BOLD}🎤 INTERACTIVE LYRICS PLAYBACK: May not work outside macOS{RESET}")
    print("=" * 70)
    print(f"  Playing: {audio_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Words: {len(words_with_timestamps)}")
    if stats:
        print(
            f"  Accuracy: {stats['accuracy']:.1f}% ({GREEN}correct{RESET} / {RED}wrong{RESET})"
        )
    print("=" * 70)
    print("\n  Press Ctrl+C to stop\n")

    # Start audio playback in background using afplay (macOS)
    # For Linux, you might use 'aplay' or 'paplay'
    audio_process = subprocess.Popen(
        ["afplay", str(audio_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    start_time = time.time()

    # Print initial content that we'll update in place
    print(f"   {format_timestamp(0)} [{'░' * 40}] {format_timestamp(duration)}")
    print(f"  ", end="", flush=True)

    try:
        while audio_process.poll() is None:
            current_time = time.time() - start_time

            if current_time > duration:
                break

            # Get current words display (with colors if alignment provided)
            lyrics_display = get_current_words(
                words_with_timestamps, current_time, alignment=alignment
            )

            # Progress bar
            progress = current_time / duration
            bar_width = 40
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Move cursor up 1 line, clear it, print progress, then clear next line and print lyrics
            sys.stdout.write("\033[1A\033[2K\r")  # Move up 1, clear line
            print(
                f"   {format_timestamp(current_time)} [{bar}] {format_timestamp(duration)}"
            )
            sys.stdout.write("\033[2K\r")  # Clear current line
            print(f"  {lyrics_display}", end="", flush=True)

            time.sleep(0.05)  # Update ~20 times per second

    except KeyboardInterrupt:
        audio_process.terminate()
        print("\n\n  Playback stopped.")

    audio_process.wait()
    print("\n\n" + "=" * 70)
    print(f"  {BOLD}Playback complete!{RESET}")
    print("=" * 70)


def transcribe_audio(
    audio_path,
    device=None,
    use_lm=False,
    lm_path=None,
    alpha=0.5,
    beta=1.0,
    with_timestamps=False,
):
    # Transcribes audio using Wav2Vec2

    if device is None:
        device = get_device()

    # Timestamps require LM decoding
    if with_timestamps and not use_lm:
        print("Note: Timestamps require language model decoding. Enabling --use-lm.")
        use_lm = True

    # Load model
    model, processor = load_wav2vec2_model()
    model.to(device)
    model.eval()

    # Load and resample audio
    print(f"Loading audio: {audio_path}")
    audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)

    print(f"Original sample rate: {orig_sr}")
    audio_16k = resample_audio(audio, orig_sr, target_sr=16000)
    print(f"Resampled to 16000 Hz, length: {len(audio_16k)} samples")

    # Get logits
    logits = get_all_logits(audio_16k, model, processor)

    # Get emission and transcription
    emission = get_emission(logits)

    # Use language model decoding if requested
    decoder = None
    words_with_timestamps = None

    if use_lm:
        if lm_path is None:
            lm_path = download_language_model()
        decoder = build_ctc_decoder(processor, lm_path, alpha, beta)

        if with_timestamps:
            transcription, words_with_timestamps = get_transcription_with_timestamps(
                logits, decoder, sample_rate=16000
            )
            print("Using language model decoding with timestamps")
        else:
            transcription = get_transcription_with_lm(logits, decoder)
            print("Using language model decoding")
    else:
        transcription = get_transcription(logits, processor)
        print("Using greedy decoding")

    return {
        "transcription": transcription,
        "words_with_timestamps": words_with_timestamps,
        "logits": logits,
        "emission": emission,
        "audio_16k": audio_16k,
        "model": model,
        "processor": processor,
        "decoder": decoder,
    }


# #endregion
# MAIN SCRIPT

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vocal separation and transcription using htdemucs and Wav2Vec2"
    )
    parser.add_argument("--audio_path", type=str, help="Path to the input audio file")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--separate-only",
        action="store_true",
        help="Only separate vocals, don't transcribe",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only transcribe audio (assumes input is already vocals)",
    )

    # Language model options
    parser.add_argument(
        "--use-lm",
        action="store_true",
        help="Use language model for better decoding (downloads model if needed)",
    )
    parser.add_argument(
        "--lm-path",
        type=str,
        default=None,
        help="Path to KenLM language model file (.arpa or .bin)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Language model weight (higher = more LM influence, default: 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Word insertion bonus (higher = more words, default: 1.0)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Output word-level timestamps (requires/enables --use-lm for wav2vec2)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Interactive karaoke-style playback with synchronized lyrics (requires/enables --timestamps)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="wav2vec2",
        choices=["wav2vec2", "parakeet"],
        help="ASR model to use: wav2vec2 (default) or parakeet (NVIDIA NeMo)",
    )

    # Noise injection
    parser.add_argument(
        "--noise",
        type=float,
        nargs="?",
        const=0.07,
        default=None,
        help="Add white noise before transcription. Optional amplitude (0.0-1.0, default: 0.07)",
    )

    # Reference lyrics for comparison
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Path to reference lyrics file for accuracy comparison",
    )

    args = parser.parse_args()

    # --play implies --timestamps
    if args.play:
        args.timestamps = True
    # For wav2vec2, --timestamps implies --use-lm
    if args.timestamps and args.model == "wav2vec2":
        args.use_lm = True

    if args.separate_only:
        result = separate_vocals_from_audio(args.audio_path, output_dir=args.output_dir)
        print("\nVocal separation complete!")

    elif args.transcribe_only:
        # Determine audio path (apply noise if requested)
        audio_to_transcribe = args.audio_path
        if args.noise is not None:
            audio_to_transcribe = add_noise_to_audio(args.audio_path, args.noise)

        # Use selected model
        if args.model == "parakeet":
            result = transcribe_with_parakeet(audio_to_transcribe)
        else:
            result = transcribe_audio(
                audio_to_transcribe,
                use_lm=args.use_lm,
                lm_path=args.lm_path,
                with_timestamps=args.timestamps,
            )

        # Compare with reference lyrics if provided
        alignment = None
        stats = None
        if args.reference and result.get("words_with_timestamps"):
            reference_words_original, reference_words_normalized = (
                load_reference_lyrics(args.reference)
            )
            transcribed_words = [w for w, _, _ in result["words_with_timestamps"]]
            alignment, stats = align_words(
                transcribed_words, reference_words_normalized, reference_words_original
            )

            # Print comparison results
            print_comparison_stats(stats)
            print_colored_transcription(alignment)

        if result.get("words_with_timestamps"):
            if not args.reference:
                print_timestamped_transcription(result["words_with_timestamps"])

            if args.play:
                input("\nPress Enter to start karaoke playback...")
                # Play the original (or noisy) audio with lyrics
                play_audio_with_lyrics(
                    audio_to_transcribe,
                    result["words_with_timestamps"],
                    alignment=alignment,
                    stats=stats,
                )
        else:
            print(f"\nTranscription:\n{result['transcription']}")

    else:
        # Full pipeline: separate vocals then transcribe

        # Step 1: Separate vocals
        separation_result = separate_vocals_from_audio(
            args.audio_path, output_dir=args.output_dir
        )

        # Get vocals path
        vocals_path = Path(args.output_dir) / f"{Path(args.audio_path).stem}_vocals.wav"

        # Apply noise to vocals if requested
        audio_to_transcribe = str(vocals_path)
        if args.noise is not None:
            audio_to_transcribe = add_noise_to_audio(str(vocals_path), args.noise)

        # Step 2: Transcribe with selected model
        print("\n" + "=" * 60)
        print("STEP 2: Transcribing vocals")
        print("=" * 60)

        if args.model == "parakeet":
            result = transcribe_with_parakeet(audio_to_transcribe)
        else:
            result = transcribe_audio(
                audio_to_transcribe,
                use_lm=args.use_lm,
                lm_path=args.lm_path,
                alpha=args.alpha,
                beta=args.beta,
                with_timestamps=args.timestamps,
            )

        # Add separation results to result dict
        result["vocals"] = separation_result["vocals"]
        result["instrumentals"] = separation_result["instrumentals"]
        result["sample_rate"] = separation_result["sample_rate"]

        # Compare with reference lyrics if provided
        alignment = None
        stats = None
        if args.reference and result.get("words_with_timestamps"):
            reference_words_original, reference_words_normalized = (
                load_reference_lyrics(args.reference)
            )
            transcribed_words = [w for w, _, _ in result["words_with_timestamps"]]
            alignment, stats = align_words(
                transcribed_words, reference_words_normalized, reference_words_original
            )

            # Print comparison results
            print_comparison_stats(stats)
            print_colored_transcription(alignment)

        if result.get("words_with_timestamps"):
            if not args.reference:
                print_timestamped_transcription(result["words_with_timestamps"])

            if args.play:
                input("\nPress Enter to start karaoke playback of vocals...")
                play_audio_with_lyrics(
                    audio_to_transcribe,
                    result["words_with_timestamps"],
                    alignment=alignment,
                    stats=stats,
                )
        else:
            print(f"\nTranscription:\n{result['transcription']}")

        print("\nProcessing complete!")
