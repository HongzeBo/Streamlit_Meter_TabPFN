# Hygro-Thermobarometer — TabPFN edition

A trial deployment of the Bo, Klein & Jagoutz hygro-thermobarometer that uses
**[TabPFN](https://github.com/PriorLabs/TabPFN)** instead of pre-trained
RandomForest models.

## Why a separate app

The original tool (`HongzeBo/Streamlit_Meter`) ships ~130 MB of RandomForest
`.pkl` files saved with **scikit-learn 1.2.0**. TabPFN requires a much newer
stack (scikit-learn ≥ 1.6, numpy ≥ 2, PyTorch), under which those old pickles
fail to load — so the two cannot share one environment. This repo is the clean
TabPFN-only counterpart.

## How it differs

- **No saved model files.** TabPFN is an in-context foundation model: for the
  liquid/mineral pair you select, it is *fit on the calibration sheet on the fly*
  and predicts immediately. The app therefore ships only the 2.7 MB calibration
  workbook (`allexps_allpairs.xlsx`), not a model zoo.
- **Default = Melt_only-PlgSat hygrometer**, the model deployed in the paper
  (calibration RMSE ≈ 1.04 wt.% H₂O; independent validation ≈ 1.24 wt.%).
- The retired hybrid RF tab is gone.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the TabPFN v2 regressor checkpoint
(~tens of MB) to a local cache; subsequent predictions are fast.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io → **New app** → pick this repo / `main` /
   `app.py`.
3. First boot is slow (PyTorch import + checkpoint download). If the build
   pulls a CUDA build of torch, confirm the `--extra-index-url .../whl/cpu`
   line is the first line of `requirements.txt`.

## Files

| file | purpose |
|------|---------|
| `app.py` | Streamlit UI + TabPFN fit/predict |
| `allexps_allpairs.xlsx` | 57 calibration sheets (index-aligned with the pair list) |
| `meter_rank.xlsx` / `meter_rank_old.xlsx` | pair display names ↔ sheet order |
| `Template_input.xlsx` | upload template (Component 1 = melt, Component 2 = mineral) |
