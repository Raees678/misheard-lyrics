# Misheard Lyrics

## Text based misheard lyrics

## Vocal Separation and Transcription with Wav2Vec2 and Parakeet

Please note that playback only works on macOS. The code is in the vocal_separation.py file.

Python version: 3.11.13

Install dependencies:

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
