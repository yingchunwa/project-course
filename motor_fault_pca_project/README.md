# Motor Fault Detection with Independent PCA Branches

This project implements a lightweight motor fault detection baseline for edge
devices such as Orange Pi. It uses two independent PCA branches:

- Visual motion features PCA
- ADXL345 three-axis vibration features PCA

There is no visual-vibration fusion, and no PyTorch or TensorFlow dependency.

Runtime hardware target:

- Camera: USB UVC global camera, `YUY2 160x140@420fps`
- Vibration: I2C motion sensor at bus `7`, address `0x6A`, `WHO_AM_I=0x70`
- Host: Orange Pi Linux

## Directory Layout

```text
motor_fault_pca_project/
├── features/
│   ├── visual_motion_features.csv
│   └── vibration_features.csv
├── models/
│   ├── visual_pca_model.pkl
│   └── vibration_pca_model.pkl
├── results/
│   ├── visual_pca_results.csv
│   └── vibration_pca_results.csv
├── src/
│   ├── pca_detector.py
│   ├── utils.py
│   ├── train_visual_pca.py
│   ├── test_visual_pca.py
│   ├── train_vibration_pca.py
│   ├── test_vibration_pca.py
│   ├── collect_visual_features.py
│   ├── collect_vibration_features.py
│   └── realtime_detect.py
└── requirements.txt
```

## Input CSV Format

Each CSV row represents one time window.

Required training label:

```text
label
```

Supported label values:

```text
normal
fault
```

Metadata columns excluded from PCA features:

```text
sample_id, run_id, window_id, start_time, end_time, label, rpm
```

All remaining numeric columns are used as PCA features.

## PCA Method

Training:

```text
StandardScaler -> PCA(n_components=0.99)
```

Only `normal` samples are used to fit the scaler and PCA. The reconstruction
error is computed on normal training samples, and the threshold is set to the
95th percentile.

Testing:

```text
score = mean squared reconstruction error
prediction = fault if score > threshold else normal
```

The result CSV saves:

```text
score, threshold, prediction
```

If `label` exists in the input CSV, the script also prints:

```text
accuracy, precision, recall, f1, confusion matrix, AUROC
```

## Install

```bash
pip install -r requirements.txt
```

For realtime camera and serial detection, install:

```bash
pip install -r requirements-realtime.txt
```

## Collect Training Data from Hardware

Collect normal visual windows:

```bash
python src/collect_visual_features.py --label normal --windows 100
```

Collect fault visual windows:

```bash
python src/collect_visual_features.py --label fault --windows 100
```

Collect normal vibration windows from the I2C sensor:

```bash
python src/collect_vibration_features.py --source i2c --label normal --windows 100
```

Collect fault vibration windows from the I2C sensor:

```bash
python src/collect_vibration_features.py --source i2c --label fault --windows 100
```

Default vibration I2C settings:

```text
i2c_bus = 7
i2c_addr = 0x6A
sample_rate_hz = 60
window_seconds = 0.25
```

Serial mode is still available if a later STM32 USB CDC bridge is used:

```bash
python src/collect_vibration_features.py --source serial --port /dev/ttyACM0 --label normal
```

If STM32 outputs sequence and timestamp first:

```text
seq,tick_us,ax,ay,az
```

then add:

```bash
--axis-start-index 2
```

## Run Full Train and Test Pipeline

After placing both CSV files under `features/`, run:

```bash
python src/run_all.py
```

This executes:

```text
train visual PCA -> test visual PCA -> train vibration PCA -> test vibration PCA
```

Expected outputs:

```text
models/visual_pca_model.pkl
models/vibration_pca_model.pkl
results/visual_pca_results.csv
results/vibration_pca_results.csv
```

## Realtime Detection

Run visual branch only:

```bash
python src/realtime_detect.py --visual
```

Run ADXL345 vibration branch only:

```bash
python src/realtime_detect.py --vibration --vibration-source i2c
```

Run both independent branches at the same time:

```bash
python src/realtime_detect.py --visual --vibration --vibration-source i2c
```

Each branch prints:

```text
score, threshold, prediction
```

The two branches remain independent. No fusion is performed.

## Periodic PCA Retraining

For realtime deployment, do not update PCA after every new window. Use a normal
buffer:

```text
realtime detection
-> save high-confidence normal windows
-> accumulate enough windows
-> retrain StandardScaler + PCA + threshold offline
-> back up old model
-> replace current model
```

Example: run vibration realtime detection and save only confident normal windows:

```bash
python src/realtime_detect.py \
  --vibration \
  --vibration-source i2c \
  --window-seconds 2 \
  --save-normal-candidates \
  --normal-candidate-ratio 0.5
```

Candidate windows are saved to:

```text
features/normal_candidates/vibration_normal_candidates.csv
features/normal_candidates/visual_normal_candidates.csv
```

Retrain vibration PCA from the latest confirmed-normal buffer:

```bash
python src/retrain_pca_from_normal_buffer.py \
  --branch vibration \
  --min-windows 100 \
  --max-windows 1000 \
  --threshold-quantile 0.99 \
  --threshold-scale 1.2
```

Retrain visual PCA:

```bash
python src/retrain_pca_from_normal_buffer.py \
  --branch visual \
  --min-windows 100 \
  --max-windows 1000 \
  --threshold-quantile 0.99 \
  --threshold-scale 1.2
```

Old models are backed up under:

```text
models/backups/
```

## Train and Test Visual PCA

Put the visual feature file here:

```text
features/visual_motion_features.csv
```

Train:

```bash
python src/train_visual_pca.py
```

Test:

```bash
python src/test_visual_pca.py
```

Outputs:

```text
models/visual_pca_model.pkl
results/visual_pca_results.csv
```

## Train and Test Vibration PCA

Put the vibration feature file here:

```text
features/vibration_features.csv
```

Train:

```bash
python src/train_vibration_pca.py
```

Test:

```bash
python src/test_vibration_pca.py
```

Outputs:

```text
models/vibration_pca_model.pkl
results/vibration_pca_results.csv
```

## Notes for Orange Pi

- Use Python 3.9+ if possible.
- Install `requirements.txt` for offline PCA training/testing.
- Install `requirements-realtime.txt` for camera and ADXL345 serial detection.
- Keep the two branches independent unless fusion is explicitly added later.
