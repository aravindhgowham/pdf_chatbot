# 1D CNN Raw-Audio PASS/FAIL Classifier (PyTorch)

This project trains a 1D CNN model directly on raw waveform audio (.wav) to classify horn sounds as PASS or FAIL.

## Setup

1. Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Configure environment defaults via `.env` in project root:

```
AUDIO_SAMPLE_RATE=16000
AUDIO_DURATION_SEC=4.0
```

3. Prepare your data directory with this structure:

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

## Training (basic)

```
python src/train.py \
  --train_dir /data/train \
  --val_dir /data/val \
  --output_dir checkpoints \
  --sample_rate 16000 \
  --duration_sec 4.0 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --num_workers 4 \
  --model_name basic
```

## Training (advanced, better accuracy)

Key options: residual model with attention, mixup, time-shift, label-smoothing, EMA, early stopping, gradient clipping.

```
python src/train.py \
  --train_dir /data/train \
  --val_dir /data/val \
  --output_dir checkpoints \
  --sample_rate 16000 \
  --duration_sec 4.0 \
  --batch_size 32 \
  --epochs 60 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --model_name resattn \
  --base_channels 64 \
  --dropout 0.2 \
  --time_shift_ms 50 \
  --mixup_alpha 0.2 \
  --label_smoothing 0.05 \
  --ema_decay 0.995 \
  --early_stop_patience 12 \
  --clip_grad_norm 2.0 \
  --use_weighted_sampler
```

This will save best model checkpoint(s) and label mapping into `checkpoints/`:
- `best_model.pt` (standard)
- `best_model_ema.pt` (EMA, often better)

## Inference on a single file

```
python src/infer.py \
  --checkpoint checkpoints/best_model_ema.pt \
  --label_map checkpoints/label_map.yaml \
  --audio_path /path/to/horn.wav \
  --sample_rate 16000 \
  --duration_sec 4.0
```

Example output:

```
Prediction: PASS (p=0.93)
```

## Notes

- Model options: `basic` (compact CNN) or `resnet`/`resattn` (deeper residual with optional attention pooling).
- Advanced training includes: mixup, time-shift, label smoothing, EMA, early stopping, gradient clipping, weighted sampling.
- Input is resampled and padded/truncated to the target duration for batch training consistency.
- Augmentation avoids additive noise by default; only light gain jitter plus optional time shift/mixup are applied during training.
- You can adjust model depth/width in `src/model_adv.py` and augmentation settings in `src/augment.py`.