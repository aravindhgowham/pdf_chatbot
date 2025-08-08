# 1D CNN Raw-Audio PASS/FAIL Classifier (PyTorch)

This project trains a 1D CNN model directly on raw waveform audio (.wav) to classify horn sounds as PASS or FAIL.

## Setup

1. Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your data directory with this structure:

```
/data
  train/
    pass/  # .wav files that are PASS
    fail/  # .wav files that are FAIL
  val/
    pass/
    fail/
```

You can optionally add a `test/` directory with the same subfolders.

WAV requirements: mono or stereo, any sample rate (will be resampled); 16-bit PCM preferred.

## Training

```
python src/train.py \
  --train_dir /data/train \
  --val_dir /data/val \
  --output_dir checkpoints \
  --sample_rate 16000 \
  --duration_sec 2.0 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --num_workers 4
```

This will save best model checkpoint and label mapping into `checkpoints/`.

## Inference on a single file

```
python src/infer.py \
  --checkpoint checkpoints/best_model.pt \
  --label_map checkpoints/label_map.yaml \
  --audio_path /path/to/horn.wav \
  --sample_rate 16000 \
  --duration_sec 2.0
```

Example output:

```
Prediction: PASS (p=0.93)
```

## Notes

- Model trains on raw waveform; no spectrograms or handcrafted features are used.
- Input is resampled and padded/truncated to the target duration for batch training consistency.
- You can adjust model depth/width in `src/model.py` and augmentation settings in `src/dataset.py`.