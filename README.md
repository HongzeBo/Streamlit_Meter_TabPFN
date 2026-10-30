# Hygro-Thermobarometer — Bo, Klein & Jagoutz

The web tool of *Water Controls on Magmatic Differentiation I: Experimental Constraints and
Melt-Composition Hygrometry* (Part I) and its companion Part II, deployed at
https://hygrothermobaro.streamlit.app.

For the liquid/mineral pair selected, a [TabPFN](https://github.com/PriorLabs/TabPFN)
regressor is fit in-context on the experimental calibration workbook and applied to the
uploaded samples, so the app stores no model files. The default is the Melt_only-PlgSat
hygrometer (random-split test RMSE ≈ 1.00 wt.% H₂O). The app also serves the melt-fraction
model of Part II (Section 5) and the plagioclase-saturation classifier of Part I (Section 7.1),
and tests every sample against the calibration domain of the selected model in the seven
SiO₂–oxide projections of the paper. This app replaced the earlier RandomForest tool
(`HongzeBo/Streamlit_Meter`).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the TabPFN v2 regressor checkpoint to a local cache;
subsequent predictions are fast.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io → **New app** → pick this repo / `main` / `app.py`.
3. First boot is slow (PyTorch import + checkpoint download). If the build pulls a CUDA
   build of torch, confirm the `--extra-index-url .../whl/cpu` line is the first line of
   `requirements.txt`.

## Files

| file | purpose |
|------|---------|
| `app.py` | Streamlit UI, TabPFN fit/predict, domain check, classifier |
| `allexps_allpairs.xlsx` | 57 pair calibration sheets (index-aligned with the pair list) + the appended `meltfrac` sheet (index 57). **The `meltfrac` sheet exists only in this deployed copy** — `Water_Hygrometer/data/experiments/allexps_allpairs.xlsx` has 57 sheets and no `meltfrac`, so copying that file over this one would silently remove the melt-fraction model. Rebuild the sheet from `Water_Hygrometer/data/ml_features/exp_meltfraction_primarymgnum60_75.xlsx` (majors renormalised to 100 wt.%, then *F* repeated in the three trailing meta columns). |
| `meter_rank.xlsx` / `meter_rank_old.xlsx` | pair display names ↔ sheet order |
| `method_comparison.xlsx` | TabPFN vs. baseline test RMSE per pair |
| `Template_input.xlsx` | upload template (Component 1 = melt, Component 2 = mineral; all iron as FeOT) |
| `assets/Template_input_plgsat.xlsx` | melt-only template (hygrometer and classifier) |
| `assets/plg-sat.xlsx` | training set of the plagioclase-saturation classifier (copy of `Water_Hygrometer/data/ml_features/plg-sat.xlsx`) |
