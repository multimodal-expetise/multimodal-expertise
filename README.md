# Multimodal Expertise — Evaluation of Pretrained Models

Reproduction code for the paper evaluating multimodal fusion models on the Expertise dataset.

## Models (14 total)

| # | Model | Modality | Fusion |
|---|-------|----------|--------|
| A1–A3 | LSTM | Single (L / A / V) | — |
| A4–A6 | LF_LSTM | Dual (LA / LV / AV) | Late |
| A7 | LF_LSTM | Triple (LAV) | Late |
| A8 | LF_MLP | Triple (LAV) | Late |
| A9 | EF_MLP | Triple (LAV) | Early |
| A10 | EF_LSTM | Triple (LAV) | Early |
| A11 | MFN w/o A↔V | Triple (LAV) | MFN |
| A12 | MFN w/o T↔A | Triple (LAV) | MFN |
| A13 | MFN w/o T↔V | Triple (LAV) | MFN |
| A14 | MFN (full) | Triple (LAV) | MFN |

L = Language/Text, A = Audio, V = Visual

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Place the test feature file at `data/expertise_test.pkl`. Expected keys:
- `text`: shape `(N, T_l, 768)`
- `audio`: shape `(N, T_a, 74)`
- `vision`: shape `(N, T_v, 35)`
- `labels`: shape `(N,)`, integer scale (shifted by −4 internally to zero-center)
- `raw_text`, `info`: metadata

## Pretrained Checkpoints

Place `.pth` state-dict files in `pretrained_model/` named:
```
{model_name}_pretrained_model.pth
```
e.g. `A1_LSTM_l_pretrained_model.pth`, `A14_MFN_lav_pretrained_model.pth`.

## Run Evaluation

```bash
python main_run_evaluation_pretrained_model.py
```

Reports `Acc_2`, `F1_score`, `MAE`, `Corr` for each model.

## Project Structure

```
├── main_run_evaluation_pretrained_model.py   # entry point
├── module_test.py                             # load checkpoint + evaluate
├── data_loader.py                             # dataset + dataloader
├── config/
│   ├── config.py                              # config loader
│   └── config_pretrained.json                 # per-model hyperparameters
├── models/
│   ├── AMIO.py                                # model registry
│   └── singleTask/                            # A1–A14 model definitions
├── utils/
│   └── metricsTop.py                          # evaluation metrics + set_seed
├── data/                                      # test features
└── pretrained_model/                          # checkpoints
```
