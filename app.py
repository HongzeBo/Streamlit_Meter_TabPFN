#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Hygro-thermobarometer web tool of Bo, Klein & Jagoutz (Part I): a TabPFN regressor is
# fit in-context on the experimental calibration workbook for the liquid/mineral pair
# selected, so no model files are stored. Also serves the melt-fraction model of Part II
# (Section 5) and the plagioclase-saturation classifier of Part I (Section 7.1).
# deploy history: 2026-10-30 classifier tab, seven-projection domain check, template
#   header FeOT, shortened Info page; 2026-08-13 melt-fraction calibration (starting
#   Mg# 60-75); 2026-07-30 per-sample prediction intervals; 2026-07-02 corrected-DB models.

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CAL_BOOK = "allexps_allpairs.xlsx"          # 57 pair sheets (index-aligned) + meltfrac
OXIDES = ["SiO$_2$", "TiO$_2$", "Al$_2$O$_3$", "FeOT", "MnO",
          "MgO", "CaO", "Na$_2$O", "K$_2$O", "P$_2$O$_5$"]

# pretty pair name (meter_rank)  <->  file/sheet pair name (meter_rank_old),
# the latter aligned by index with the calibration workbook's sheet order.
TARGETS = {
    "Water (wt.%)":       dict(key="hygro", col=-3, unit="H$_2$O (wt.%)"),
    "Temperature (°C)":   dict(key="thermo", col=-1, unit="T (°C)"),
    "Pressure (kbar)":    dict(key="baro", col=-2, unit="P (kbar)"),
    "Melt fraction (melt-only)": dict(key="meltfrac", col=-1, unit="Melt fraction"),
}
# Appended calibration sheet (847 rows x 13 cols): 10 melt majors renormalised to
# 100 wt.%, then the cumulative melt fraction F repeated in all three trailing
# meta slots, so the generic [H2O, P, T] slicing below picks F at col = -1.
# Source: experimental series whose parental melt has a starting Mg# of 60-75
# (Water_Hygrometer/data/ml_features/exp_meltfraction_primarymgnum60_75.xlsx,
#  126 series).  This sheet exists ONLY in the deployed copy of the workbook:
#  Water_Hygrometer/data/experiments/allexps_allpairs.xlsx has 57 sheets and no
#  meltfrac sheet, so never overwrite this file with that one wholesale.
MELTFRAC_SHEET = 57

# Percentiles of the TabPFN predictive distribution reported for the deployed
# Melt_only-PlgSat hygrometer (paper Section 6.1): mean = point prediction,
# 16th-84th = 68% prediction interval, 5th-95th = 90% prediction interval.
HYGRO_QUANTILES = [0.05, 0.16, 0.50, 0.84, 0.95]

# Calibration-domain check (paper Section 7.1): convex hulls of the calibration set in the
# seven SiO2-vs-oxide projections TiO2, Al2O3, FeOT, MgO, CaO, Na2O, K2O (MnO and P2O5 are
# too often unreported to define a domain). Indices into a 10-oxide block.
DOMAIN_PROJ = [1, 2, 3, 5, 6, 7, 8]

# Plagioclase-saturation classifier (paper Section 7.1; the screen used for the natural
# samples of Part II): RandomForest(64 trees, class-weighted, random_state=0) retrained
# deterministically from the training workbook (10 raw melt majors, label in the last column).
# Compositions with SiO2 >= 80 or MgO >= 15 wt.% are classed as not Plg-saturated outright.
PLGSAT_XLSX = "assets/plg-sat.xlsx"
CLF_SIO2_MAX, CLF_MGO_MAX = 80.0, 15.0


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
def load_comparison():
    """Per-pair test RMSE of TabPFN vs linear / kNN / RF / XGBoost, by target."""
    xl = pd.ExcelFile("method_comparison.xlsx")
    return {s: xl.parse(s) for s in xl.sheet_names}


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


@st.cache_resource(show_spinner=False)
def get_plg_classifier():
    """Deterministic retrain of the paper's Plg-saturation classifier (Section 7.1)."""
    from sklearn.ensemble import RandomForestClassifier
    data = pd.read_excel(PLGSAT_XLSX, sheet_name=0, header=None).to_numpy(float)
    X, y = data[:, :-4], data[:, -1]
    clf = RandomForestClassifier(n_estimators=64, class_weight="balanced",
                                 random_state=0).fit(X, y)
    return clf, int(len(y))


def classify_plgsat(X10):
    """Apply the classifier as in the paper: majors renormalised to 100 wt.%, then
    SiO2 >= 80 or MgO >= 15 wt.% -> not Plg-saturated regardless of the vote.
    Returns (plg_saturated int[n], P(Plg-saturated) float[n], screened bool[n]).
    """
    clf, _ = get_plg_classifier()
    X = renormalize(np.asarray(X10, float))
    prob = clf.predict_proba(X)[:, list(clf.classes_).index(1.0)]
    sat = clf.predict(X).astype(int)
    screened = (X[:, 0] >= CLF_SIO2_MAX) | (X[:, 5] >= CLF_MGO_MAX)
    sat[screened] = 0
    return sat, prob, screened


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


def domain_flags(cal_X, data_in):
    """Per-sample in-/out-of-domain test against the calibration convex hull.

    For every block present in the upload (melt majors at columns 0-9, and the
    mineral majors at 10-19 when a second component is supplied) we build the
    SiO2-vs-oxide convex hull of the calibration set for the seven projections
    of ``DOMAIN_PROJ`` (paper Section 7.1), and test each input row against all
    of them. A sample is in-domain only when it lies inside every projection's
    hull; otherwise the prediction is an extrapolation. Returns
    ``(in_domain bool[n], n_out int[n])`` where ``n_out`` counts how many
    projections a sample fell outside of.
    """
    from matplotlib.path import Path as _Path
    n = len(data_in)
    try:
        from scipy.spatial import ConvexHull
    except Exception:                                   # scipy absent -> skip check
        return np.ones(n, bool), np.zeros(n, int)

    inside = np.ones(n, bool)
    n_out = np.zeros(n, int)
    blocks = [0] if cal_X.shape[1] == 10 else [0, 10]
    for off in blocks:
        for k in DOMAIN_PROJ:
            cols = [off, off + k]
            pts = cal_X[:, cols]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 4:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:                           # degenerate (collinear) -> skip
                continue
            poly = _Path(pts[hull.vertices])
            q = data_in[:, cols]
            # OR both winding orientations so boundary points count as in-domain
            isin = (poly.contains_points(q, radius=1e-9)
                    | poly.contains_points(q, radius=-1e-9))
            n_out += (~isin).astype(int)
            inside &= isin
    return inside, n_out


def harker_figure(data_in, y_pred, cal_X, unit, block_offset=0, in_dom=None):
    """3x3 Harker panel: SiO2 vs the other 9 oxides, calibration grey + input colour.

    Out-of-domain samples (``in_dom`` False) are over-marked with a red cross.
    """
    fig = plt.figure(figsize=(10, 7.5))
    cmap, alpha = "viridis", 0.3
    xin = data_in[:, block_offset]
    xcal = cal_X[:, block_offset]
    out = (~np.asarray(in_dom, bool)) if in_dom is not None else None
    for k in range(9):
        ax = fig.add_subplot(3, 3, k + 1)
        ax.scatter(xcal, cal_X[:, block_offset + k + 1], s=12, linewidths=0,
                   color="grey", alpha=alpha, label="Calibration")
        sc = ax.scatter(xin, data_in[:, block_offset + k + 1], c=y_pred,
                        cmap=cmap, edgecolors="black", linewidths=0.2,
                        label="Input")
        if out is not None and out.any():
            ax.scatter(xin[out], data_in[out, block_offset + k + 1], s=70,
                       marker="x", color="red", linewidths=1.1, zorder=6,
                       label="Out of domain")
        fig.colorbar(sc, ax=ax, label=unit)
        ax.set_xlabel(OXIDES[0], fontweight="bold")
        ax.set_ylabel(OXIDES[k + 1], fontweight="bold")
    fig.tight_layout()
    return fig


def build_output_xlsx(data_in, y_pred, target_unit, in_dom=None,
                      quantiles=None) -> bytes:
    cols = [o.replace("$", "").replace("_", "") for o in OXIDES]
    if data_in.shape[1] == 20:
        cols = [f"C1_{c}" for c in cols] + [f"C2_{c}" for c in cols]
    df = pd.DataFrame(data_in, columns=cols[:data_in.shape[1]])
    df[target_unit.split(" ")[0] + "_pred"] = y_pred
    if quantiles is not None:
        # PlgSat hygrometer: percentiles of the TabPFN predictive distribution
        # (point prediction = distribution mean; q16-q84 = 68%, q05-q95 = 90% prediction
        # interval; meaningful only for in-domain samples).
        for name, vals in quantiles.items():
            df[name] = vals
    if in_dom is not None:
        df["in_domain"] = np.where(np.asarray(in_dom, bool),
                                   "yes", "no (extrapolation)")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Predictions")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hygro-Thermobarometer", layout="wide")
# Standard web layout: cap the content to a centered max-width column so wide
# screens get whitespace on both sides (instead of full-bleed edge-to-edge).
st.markdown(
    "<style>.block-container{max-width:1080px;margin:0 auto;"
    "padding-top:2.5rem;padding-left:2.5rem;padding-right:2.5rem;}</style>",
    unsafe_allow_html=True,
)
st.title("Hygro-Thermobarometer Based on a Tabular Foundation Model — Bo, Klein & Jagoutz")

tab_calc, tab_clf, tab_info = st.tabs(["**Calc & Ranking**", "**Plg-saturation classifier**", "**Info**"])

with tab_calc:
    rank, old = load_rank_tables()

    st.write("***Rankings of hygrometers and thermobarometers — test RMSE per pair, "
             "TabPFN vs. baseline methods on the same 20× 80:20 random splits. TabPFN "
             "gives the lowest test error for every pair.***")
    comp = load_comparison()
    which = st.radio("Show comparison for:", list(comp.keys()),
                     horizontal=True, key="cmp_target")
    st.dataframe(comp[which], hide_index=True)

    st.subheader("***Step 1: What do you want to predict?***")
    q_target = st.selectbox(" ", list(TARGETS.keys()), key="target")
    tgt = TARGETS[q_target]

    if tgt["key"] == "meltfrac":
        # Melt fraction is a single melt-only model, calibrated on the experimental
        # series with a starting Mg# of 60-75 (not on all melts); no pair to choose.
        sheet_index = MELTFRAC_SHEET
        file_pair = "meltfrac"
        sel = "melt-fraction (primitive series, starting Mg# 60–75, melt-only)"
        is_plgsat_hygro = False
        st.info("**Melt-fraction model** — predicts the cumulative melt fraction "
                "$F$ (0–1) from melt major elements alone, trained on experimental "
                "series that begin from near-primitive melts (starting Mg# 60–75; "
                "847 melts from 126 series; held-out test RMSE ≈ 0.07 in $F$, "
                "R² ≈ 0.92). Fill **Component 1 (melt) only**.")
    else:
        st.subheader("***Step 2: Which liquid/mineral pair?***")
        # Map each displayed (pretty) pair to its calibration SHEET index by name,
        # via the canonical sheet-order list -- meter_rank row order is NOT sheet order.
        sheet_names = sheet_order_names()
        options = []                                    # (pretty, file_pair, sheet_index)
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
                    "(random-split test RMSE ≈ 1.00 wt.% H₂O). Apply it only to **Plg-saturated** "
                    "melts. Establish Plg saturation from petrological evidence where possible; "
                    "where none exists, the classifier tab screens melts by composition "
                    "(accuracy ≈ 86%).")

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
                    quants = None
                    if is_plgsat_hygro:
                        # Deployed hygrometer (paper Section 6.1): the point
                        # prediction stays the distribution MEAN (identical to all
                        # previously served/published predictions, incl. Part II);
                        # a second call returns the 5/16/50/84/95th percentiles of
                        # the TabPFN predictive distribution (16th-84th = 68% and
                        # 5th-95th = 90% prediction intervals). H2O is non-negative,
                        # so clamp at zero (never shrinks coverage).
                        y_pred = np.clip(np.asarray(reg.predict(data_in), float),
                                         0.0, None)
                        qraw = reg.predict(data_in, output_type="quantiles",
                                           quantiles=HYGRO_QUANTILES)
                        q05, q16, q50, q84, q95 = [
                            np.clip(np.asarray(v, float), 0.0, None) for v in qraw]
                        quants = {"H2O_q05": q05, "H2O_q16": q16,
                                  "H2O_q50_median": q50,
                                  "H2O_q84": q84, "H2O_q95": q95}
                    else:
                        y_pred = np.asarray(reg.predict(data_in), float)
                        # H2O and pressure are non-negative physical quantities; the
                        # regressor occasionally returns tiny negative values (down to
                        # ~-0.01) for genuinely dry/low-P samples. Clamp those to zero.
                        if tgt["key"] in ("hygro", "baro"):
                            y_pred = np.clip(y_pred, 0.0, None)
                        elif tgt["key"] == "meltfrac":
                            y_pred = np.clip(y_pred, 0.0, 1.0)
                st.success(f"Done — predicted {q_target} for {len(y_pred)} samples "
                           f"(TabPFN fit on {n_cal} calibration experiments).")

                # Per-model applicability: convex hull of the calibration set in
                # the SiO2-vs-oxide projections (and the mineral block, if used).
                in_dom, _n_out = domain_flags(cal_X, data_in)
                n_outside = int((~in_dom).sum())
                if n_outside:
                    st.warning(
                        f"⚠️ {n_outside} of {len(data_in)} sample(s) fall **outside "
                        f"the calibration convex hull** in one or more of the seven SiO₂–oxide "
                        f"projections — those predictions are extrapolations and less "
                        f"reliable. They are flagged in the output file's `in_domain` "
                        f"column and over-marked with a red ✕ in the Harker panel.")
                else:
                    st.info("✓ All samples lie within the calibration convex hull "
                            "(in-domain).")

                if is_plgsat_hygro:
                    st.info(
                        "**Per-sample prediction intervals** — the output file "
                        "reports the 5th/16th/50th/84th/95th percentiles of the "
                        "TabPFN predictive distribution for every sample "
                        "(the reported prediction is the distribution mean; 16th–84th = 68% interval; "
                        "5th–95th = **90% prediction interval**). Intervals are "
                        "meaningful only within the calibration domain "
                        "(`in_domain = yes`); their width tracks the local density "
                        "of calibration data.")

                c1, c2, c3 = st.columns(3)
                c1.metric(f"min {tgt['unit']}", f"{np.nanmin(y_pred):.2f}")
                c2.metric("mean", f"{np.nanmean(y_pred):.2f}")
                c3.metric("max", f"{np.nanmax(y_pred):.2f}")

                st.download_button(
                    "**Download predictions (.xlsx)**",
                    build_output_xlsx(data_in, y_pred, tgt["unit"], in_dom,
                                      quantiles=quants),
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
                st.pyplot(harker_figure(data_in, y_pred, cal_X, tgt["unit"], 0, in_dom))
                if need_w == 20:
                    st.write("**Figure 3: Major-element covariations (Component 2)**")
                    st.pyplot(harker_figure(data_in, y_pred, cal_X, tgt["unit"], 10, in_dom))

with tab_clf:
    st.write("***Plagioclase-saturation classifier (Part I, Section 7.1)***")
    st.markdown(
        "The Melt_only-PlgSat hygrometer applies only to melts that have reached plagioclase "
        "saturation. We recommend establishing this from petrological evidence, such as "
        "plagioclase phenocrysts or microlites or a plagioclase-bearing cotectic assemblage. "
        "Where no such evidence exists, this classifier predicts Plg saturation from the melt "
        "major elements alone, with the random-forest model that screened the natural samples "
        "of Part II (accuracy ≈ 86% on random splits of the experimental database, ≈ 65% when "
        "whole studies are held out). It is not recommended as a determination of Plg "
        "saturation for an individual sample, but it is a useful filter when cleaning a large "
        "dataset. Compositions with SiO₂ ≥ 80 wt.% or MgO ≥ 15 wt.% are always classed as not "
        "Plg-saturated.")

    st.subheader("***Step 1: Download the input template (melt only)***")
    with open("assets/Template_input_plgsat.xlsx", "rb") as fh:
        st.download_button("**Template_input_plgsat.xlsx**", fh,
                           file_name="Template_input_plgsat.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="dl_clf_template")

    st.subheader("***Step 2: Upload your data (Component 1 only)***")
    up_c = st.file_uploader("  ", type=["xlsx"], key="upload_clf")

    if up_c is not None:
        try:
            data_c = read_template(up_c)
        except Exception as e:                      # noqa: BLE001
            st.error(f"Could not read the upload — please use the template. ({e})")
            data_c = None

        if data_c is not None and len(data_c):
            if data_c.shape[1] != 10:
                st.error("The classifier uses the melt composition only. Fill Component 1 and "
                         "leave Component 2 blank.")
            else:
                with st.spinner("Training the classifier on the experimental database and classifying…"):
                    sat, prob, screened = classify_plgsat(data_c)
                    _, n_train = get_plg_classifier()
                n = len(sat)
                st.success(f"Done — {int(sat.sum())} of {n} samples classed as Plg-saturated "
                           f"(classifier trained on {n_train} experimental melts).")
                c1, c2, c3 = st.columns(3)
                c1.metric("Plg-saturated", int(sat.sum()))
                c2.metric("not Plg-saturated", int(n - sat.sum()))
                c3.metric("outside screening bounds", int(screened.sum()))

                cols = [o.replace("$", "").replace("_", "") for o in OXIDES]
                out = pd.DataFrame(renormalize(data_c), columns=cols)
                out["Plg_saturated"] = np.where(sat == 1, "yes", "no")
                out["P_Plg_saturated"] = np.round(prob, 3)
                out["outside_screening_bounds"] = np.where(screened, "yes", "no")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    out.to_excel(w, index=False, sheet_name="Classification")
                st.download_button("**Download classification (.xlsx)**", buf.getvalue(),
                                   file_name="PlgSat_classification.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="dl_clf_result")

                figc = plt.figure(figsize=(10, 2))
                plt.hist(prob, bins=20, range=(0, 1), color="wheat", edgecolor="black", linewidth=0.9)
                plt.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
                plt.xlabel("P(Plg-saturated)", fontweight="bold")
                plt.ylabel("Frequency", fontweight="bold")
                st.write("**Distribution of the classifier probability (dashed line = decision threshold)**")
                st.pyplot(figc)

with tab_info:
    st.markdown(
        "**Reference.** Bo, H., Klein, B. Z. & Jagoutz, O., *Water Controls on Magmatic "
        "Differentiation I: Experimental Constraints and Melt-Composition Hygrometry* (Part I) "
        "and its companion paper on global H₂O distributions (Part II). The models, their "
        "benchmarks and the sensitivity tests are described there; this page only says how to "
        "use the tool.\n\n"
        "**What it does.** For the liquid/mineral pair you select, a "
        "[TabPFN](https://github.com/PriorLabs/TabPFN) regressor is fit in-context on the "
        "experimental calibration compilation and applied to your samples. The default is the "
        "Melt_only-PlgSat hygrometer (random-split test RMSE ≈ 1.00 wt.% H₂O), which needs only "
        "the melt major elements. The melt-fraction model (Part II, Section 5) and the "
        "plagioclase-saturation classifier (accuracy ≈ 86%) have their own entries.\n\n"
        "**How to use it.** (1) Choose the target and the pair. (2) Download the template and "
        "fill the ten major oxides in wt.%, with all iron as FeOT; fill Component 2 only for "
        "mineral-bearing pairs. (3) Upload the file; the oxides are renormalised to 100 wt.% on "
        "an anhydrous basis. (4) Download the predictions. The Melt_only-PlgSat hygrometer also "
        "reports per-sample 68% and 90% prediction intervals in the output file.\n\n"
        "**Limitations.** Use the Melt_only-PlgSat hygrometer only for melts that have reached "
        "plagioclase saturation, established from petrological evidence where possible (see the "
        "classifier tab). Every sample is tested against the calibration domain of the selected "
        "model in seven SiO₂–oxide projections; samples flagged `in_domain = no` are "
        "extrapolations and should not be used. The models are calibrated on experimental "
        "glasses, so the bulk compositions of phenocryst-rich rocks and of mixed magmas carry "
        "additional bias (Part I, Sections 7.3–7.4). Whether a sample is suitable remains the "
        "user's judgement.\n\n"
        "Source code: https://github.com/HongzeBo/Streamlit_Meter_TabPFN")
