# Misheard Lyrics

## Text based misheard lyrics

Python vesion: 3.13.5

Install dependencies (make sure to use a different virtual environment for each section):

```bash
pip install -r requirements_PM.txt
```

To run the script phonetic matching:

```bash
python phonetic_matching.py --replace [song_lyrics].txt --noun-limit [maximum number of nouns to choose from for replacement]
```

Example usage:

```bash
python phonetic_matching.py --replace pioneer-lyrics.txt --noun-limit 500000
```


## Vocal Separation and Transcription with Wav2Vec2 and Parakeet

Please note that playback only works on macOS. The code is in the vocal_separation.py file.

Python version: 3.11.13

Install dependencies (make sure to use a different virtual environment for each section):

```bash
pip install -r requirements.txt
```

Wav2vec2 with vocal separation and language model. This will download the language model if it doesn't exist so be prepared to wait a few minutes.

```bash
python3 vocal_separation.py --audio_path pioneer-freddie.wav --model wav2vec2 --use-lm --play --reference pioneer-lyrics.txt
```

Wav2vec2 without vocal separation and language model. This will download the language model if it doesn't exist so be prepared to wait a few minutes.

```bash
python3 vocal_separation.py --audio_path pioneer-freddie.wav --model wav2vec2 --use-lm --play --reference pioneer-lyrics.txt --transcribe-only
```

Parakeet with vocal separation and no noise. This will download parakeet things if they dont exist.

```bash
python3 vocal_separation.py --audio_path pioneer-freddie.wav --model parakeet --play --reference pioneer-lyrics.txt
```

Parakeet without vocal separation and no noise:

```bash
python3 vocal_separation.py --audio_path pioneer-freddie.wav --model parakeet --transcribe-only --play
```

Parakeet with vocal separation and 50% noise:

```bash
python3 vocal_separation.py --audio_path pioneer-freddie.wav --model parakeet --play --reference pioneer-lyrics.txt --noise 0.5
```
