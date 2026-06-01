#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TabPFN port of the Hygro-Thermobarometer web tool.
# Original RF app: Bo, Klein & Jagoutz (2025), HongzeBo/Streamlit_Meter.
#
# What changed vs. the RF app:
#   * The prediction engine is TabPFN (in-context foundation model) instead of
#     pre-trained RandomForest .pkl files. TabPFN needs no saved model: for the
#     liquid/mineral pair you pick, it is *fit on the calibration sheet on the fly*
#     and predicts immediately. This removes the 130+ MB of sklearn-1.2.0 .pkl
#     models (and the sklearn-version wall) -- the app ships only the 2.7 MB
#     calibration workbook.
#   * The deployed headline model of the paper -- Melt_only-PlgSat hygrometer --
#     is the default selection.
#   * The hybrid RF tab was dropped (the trend+residual hybrid was retired).

import io
import os
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CAL_BOOK = "allexps_allpairs.xlsx"          # 57 calibration sheets (index-aligned)
OXIDES = ["SiO$_2$", "TiO$_2$", "Al$_2$O$_3$", "FeOT", "MnO",
          "MgO", "CaO", "Na$_2$O", "K$_2$O", "P$_2$O$_5$"]

# pretty pair name (meter_rank)  <->  file/sheet pair name (meter_rank_old),
# the latter aligned by index with the calibration workbook's sheet order.
TARGETS = {
    "Water (wt.%)":       dict(key="hygro", col=-3, unit="H$_2$O (wt.%)"),
    "Temperature (°C)":   dict(key="thermo", col=-1, unit="T (°C)"),
    "Pressure (kbar)":    dict(key="baro", col=-2, unit="P (kbar)"),
}


def sheet_order_names():
    """Canonical file-style pair names in calibration-workbook SHEET order.

    Built exactly like the original RF app so a pair name maps to the right
    sheet index regardless of how meter_rank.xlsx happens to be sorted.
    12 single-phase sheets + 45 pair sheets = 57, index-aligned with the workbook.
    """
    names = ["liq-olpyxspplgsat", "liq-pyxspplgsat", "liq-plgsat",
             "ol", "cpx", "plg", "opx", "amph", "ilm", "mag", "sp", "grt"]
    raw = ["liq", "ol", "cpx", "plg", "opx", "amph", "ilm", "mag", "sp", "grt"]
    for i in range(10):
        for j in range(10):
            if j > i:
                names.append(f"{raw[i]}-{raw[j]}")
    return names


# ----------------------------------------------------------------------------
# Data / model helpers  (cached so a re-run never re-reads or re-fits)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_rank_tables():
    rank = pd.read_excel("meter_rank.xlsx", sheet_name="Sheet1")
    old = pd.read_excel("meter_rank_old.xlsx", sheet_name="Sheet1")
    # Drop the blank separator row (Pairs == NaN) carried over from the source
    # workbook so the table and its Arrow serialization stay clean.
    keep = rank["Pairs"].notna() & old["Pairs"].notna()
    return rank[keep].reset_index(drop=True), old[keep].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_calibration(sheet_index: int):
    """Return (X_cal, Y_cal_full) for a calibration sheet.

    X_cal : renormalised feature block (10 cols for single-phase sheets,
            20 cols for pair sheets).
    Y_cal_full : the trailing 3 meta columns [H2O, P, T] (un-normalised).
    """
    data = pd.read_excel(CAL_BOOK, sheet_name=sheet_index, header=None).to_numpy(float)
    X = data[:, :-3]
    meta = data[:, -3:]
    X = renormalize(X)
    return X, meta


@st.cache_resource(show_spinner=False)
def get_tabpfn(sheet_index: int, target_col: int):
    """Fit a TabPFN regressor on (sheet_index, target_col). Cached across reruns."""
    X, meta = load_calibration(sheet_index)
    y = meta[:, target_col]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True,
                          random_state=42)
    reg.fit(X[ok], y[ok])
    return reg, int(ok.sum())


def renormalize(X: np.ndarray) -> np.ndarray:
    """Renormalise each 10-oxide block to sum 100 wt.% (10- or 20-col input)."""
    X = np.asarray(X, float).copy()
    w = X.shape[1]
    if w == 10:
        s = X.sum(axis=1, keepdims=True)
        X = np.divide(X, s, out=np.zeros_like(X), where=s > 0) * 100.0
    elif w == 20:
        for a, b in ((0, 10), (10, 20)):
            s = X[:, a:b].sum(axis=1, keepdims=True)
            X[:, a:b] = np.divide(X[:, a:b], s, out=np.zeros_like(X[:, a:b]),
                                  where=s > 0) * 100.0
    return X


def read_template(uploaded) -> np.ndarray:
    """Parse an uploaded Template_input.xlsx into a numeric matrix.

    Layout: 2 header rows, then data. Cols 0-9 = Component 1 (melt) majors,
    cols 10-19 = optional Component 2 (mineral) majors. Returns a 10- or
    20-wide float matrix with all-NaN rows/blocks dropped.
    """
    raw = pd.read_excel(uploaded, header=None, engine="openpyxl")
    body = raw.iloc[2:].apply(pd.to_numeric, errors="coerce")
    comp1 = body.iloc[:, 0:10]
    comp2 = body.iloc[:, 10:20] if body.shape[1] >= 20 else None
    use_comp2 = comp2 is not None and comp2.notna().any().any()
    block = pd.concat([comp1, comp2], axis=1) if use_comp2 else comp1
    block = block.dropna(how="all")
    block = block[comp1.loc[block.index].notna().all(axis=1)]   # need full melt row
    return block.to_numpy(float)


def harker_figure(data_in, y_pred, cal_X, unit, block_offset=0):
    """3x3 Harker panel: SiO2 vs the other 9 oxides, calibration grey + input colour."""
    fig = plt.figure(figsize=(10, 7.5))
    cmap, alpha = "hsv", 0.3
    xin = data_in[:, block_offset]
    xcal = cal_X[:, block_offset]
    for k in range(9):
        ax = fig.add_subplot(3, 3, k + 1)
        ax.scatter(xcal, cal_X[:, block_offset + k + 1], s=12, linewidths=0,
                   color="grey", alpha=alpha, label="Calibration")
        sc = ax.scatter(xin, data_in[:, block_offset + k + 1], c=y_pred,
                        cmap=cmap, edgecolors="black", linewidths=0.2,
                        label="Input")
        fig.colorbar(sc, ax=ax, label=unit)
        ax.set_xlabel(OXIDES[0], fontweight="bold")
        ax.set_ylabel(OXIDES[k + 1], fontweight="bold")
    fig.tight_layout()
    return fig


def build_output_xlsx(data_in, y_pred, target_unit) -> bytes:
    cols = [o.replace("$", "").replace("_", "") for o in OXIDES]
    if data_in.shape[1] == 20:
        cols = [f"C1_{c}" for c in cols] + [f"C2_{c}" for c in cols]
    df = pd.DataFrame(data_in, columns=cols[:data_in.shape[1]])
    df[target_unit.split(" ")[0] + "_pred"] = y_pred
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Predictions")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hygro-Thermobarometer (TabPFN)", layout="wide")
st.title("Hygro-Thermobarometer (TabPFN) — Bo, Klein & Jagoutz")
st.caption("TabPFN edition · the deployed Melt_only-PlgSat hygrometer and every "
           "liquid/mineral pair are fit in-context on the calibration data — no "
           "pre-trained model files.")

tab_calc, tab_info = st.tabs(["**Calc & Ranking**", "**Info**"])

with tab_calc:
    rank, old = load_rank_tables()
    st.write("***TabPFN test RMSE of each hygro-/thermo-/barometer pair "
             "(20× 80:20 random splits, N ≥ 200 pairs). H₂O in wt.%, T in °C, "
             "P in kbar — the same numbers as the paper's TabPFN Tables 2–4.***")
    st.dataframe(rank, hide_index=True)

    st.subheader("***Step 1: What do you want to predict?***")
    q_target = st.selectbox(" ", list(TARGETS.keys()), key="target")
    tgt = TARGETS[q_target]

    st.subheader("***Step 2: Which liquid/mineral pair?***")
    # Map each displayed (pretty) pair to its calibration SHEET index by name,
    # via the canonical sheet-order list -- meter_rank row order is NOT sheet order.
    sheet_names = sheet_order_names()
    options = []                                        # (pretty, file_pair, sheet_index)
    for pretty, file_pair in zip(rank["Pairs"], old["Pairs"]):
        if isinstance(file_pair, str) and file_pair in sheet_names:
            options.append((pretty, file_pair, sheet_names.index(file_pair)))
    pretty_list = [o[0] for o in options]
    default_idx = pretty_list.index("melt_only-plgsat") if "melt_only-plgsat" in pretty_list else 0
    sel = st.selectbox("  ", pretty_list, index=default_idx, key="pair")
    _, file_pair, sheet_index = options[pretty_list.index(sel)]

    is_plgsat_hygro = (file_pair == "liq-plgsat") and (tgt["key"] == "hygro")
    if is_plgsat_hygro:
        st.info("**Melt_only-PlgSat hygrometer** — the model deployed in the paper "
                "(calibration RMSE ≈ 1.04 wt.%). Apply it only to **Plg-saturated** "
                "melts: any melt with SiO₂ > 60 wt.%, or melts classified Plg-saturated "
                "for SiO₂ < 60 wt.%.")

    st.subheader("***Step 3: Download the input template***")
    with open("Template_input.xlsx", "rb") as fh:
        st.download_button("**Template_input.xlsx**", fh, file_name="Template_input.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("***Step 4: Upload your data***")
    up = st.file_uploader(" ", type=["xlsx"], key="upload")

    if up is not None:
        try:
            data_in = read_template(up)
        except Exception as e:                      # noqa: BLE001
            st.error(f"Could not read the upload — please use the template. ({e})")
            data_in = None

        if data_in is not None and len(data_in):
            cal_X, _ = load_calibration(sheet_index)
            need_w = cal_X.shape[1]
            if data_in.shape[1] != need_w:
                st.error(f"The **{sel}** model expects {need_w} oxide columns "
                         f"({'melt only' if need_w == 10 else 'melt + mineral'}), "
                         f"but the upload has {data_in.shape[1]}. Fill "
                         f"{'Component 1 only' if need_w == 10 else 'Components 1 and 2'}.")
            else:
                data_in = renormalize(data_in)
                with st.spinner("Fitting TabPFN on the calibration data and predicting…"):
                    reg, n_cal = get_tabpfn(sheet_index, tgt["col"])
                    y_pred = np.asarray(reg.predict(data_in), float)
                st.success(f"Done — predicted {q_target} for {len(y_pred)} samples "
                           f"(TabPFN fit on {n_cal} calibration experiments).")

                c1, c2, c3 = st.columns(3)
                c1.metric(f"min {tgt['unit']}", f"{np.nanmin(y_pred):.2f}")
                c2.metric("mean", f"{np.nanmean(y_pred):.2f}")
                c3.metric("max", f"{np.nanmax(y_pred):.2f}")

                st.download_button(
                    "**Download predictions (.xlsx)**",
                    build_output_xlsx(data_in, y_pred, tgt["unit"]),
                    file_name="TabPFN_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.write("***Grey = experimental calibration data; coloured = your "
                         "samples (colour-coded by prediction). Predictions for "
                         "inputs outside the grey field are extrapolations and less "
                         "reliable.***")
                fig1 = plt.figure(figsize=(10, 2))
                plt.hist(y_pred, color="wheat", edgecolor="black", linewidth=0.9)
                plt.xlabel(tgt["unit"], fontweight="bold")
                plt.ylabel("Frequency", fontweight="bold")
                st.write("**Figure 1: Distribution of predicted values**")
                st.pyplot(fig1)

                st.write("**Figure 2: Major-element covariations (Component 1)**")
                st.pyplot(harker_figure(data_in, y_pred, cal_X, tgt["unit"], 0))
                if need_w == 20:
                    st.write("**Figure 3: Major-element covariations (Component 2)**")
                    st.pyplot(harker_figure(data_in, y_pred, cal_X, tgt["unit"], 10))

with tab_info:
    st.subheader("Hongze Bo, Ben Klein & Oliver Jagoutz — "
                 "*Effect of H₂O on Magmatic Evolution Systematics* (in revision).")
    st.markdown(
        "- **Engine:** [TabPFN](https://github.com/PriorLabs/TabPFN) regressor, "
        "fit in-context on the experimental calibration compilation at request time.\n"
        "- **Headline model:** Melt_only-PlgSat hygrometer (calibration RMSE ≈ 1.04 wt.% "
        "H₂O; independent arc-differentiation validation RMSE ≈ 1.24 wt.%).\n"
        "- **Other pairs** (thermometry, barometry, mineral pairs) are provided for "
        "exploration; only the Melt_only family was characterised in the paper.\n"
        "- This is a **TabPFN trial deployment**; the RandomForest tool remains at "
        "the original app.")
