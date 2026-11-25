"""Tab 2: Eksplorasi Dataset, Matriks 29 Fitur, Statistik & Diagnostik CAMS (Gambar 3.7)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.features import FEATURE_GROUPS, TARGET, ChronologicalSplits


def render_dataset_tab(
    df_feat: pd.DataFrame,
    splits: ChronologicalSplits,
    split_meta: dict,
    cams_diag: pd.DataFrame,
) -> None:
    """Render chronological split cards, interactive feature dataframe, stats, and CAMS test."""
    s_info = split_meta["splits"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Dataset Bersih</div>
                <div class="kpi-value">{split_meta['clean_rows']:,} Baris × {split_meta['feature_cols'] + 1} Kolom</div>
                <div class="kpi-foot">29 Fitur Prediktor + Target pm2_5 (6 baris lag awal dibuang)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        tr, va = s_info["Train"], s_info["Validation"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Segmen Latih & Validasi</div>
                <div class="kpi-value">Train: {tr['rows']:,} | Val: {va['rows']:,}</div>
                <div class="kpi-foot">Rasio 70% : 15% · Rerata PM2.5: {tr['mean_pm25']} & {va['mean_pm25']} µg/m³</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        te = s_info["Test"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Segmen Uji (Out-of-Sample)</div>
                <div class="kpi-value">Test: {te['rows']:,} Baris (15%)</div>
                <div class="kpi-foot">Evaluasi Kronologis Buta · Rerata PM2.5: {te['mean_pm25']} µg/m³</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    st.subheader("Tabel Grid Interaktif: Matriks 29 Fitur Hasil Rekayasa + Target")

    f1, f2, f3, f4 = st.columns([1.1, 1.4, 0.9, 0.9])
    with f1:
        seg_choice = st.selectbox(
            "Segmen Partisi",
            ["Data Uji (Test - 3.945)", "Data Validasi (Val - 3.945)", "Data Latih (Train - 18.408)", "Seluruh Dataset (26.298)"],
        )
    with f2:
        grp_choice = st.selectbox(
            "Kelompok Fitur",
            ["Semua (Target + 29 Fitur)", *FEATURE_GROUPS.keys()],
        )
    with f3:
        max_rows = st.selectbox("Batas Baris Tampilan", [200, 500, 1500], index=0)

    if seg_choice.startswith("Data Uji"):
        df_seg = splits.test
    elif seg_choice.startswith("Data Validasi"):
        df_seg = splits.val
    elif seg_choice.startswith("Data Latih"):
        df_seg = splits.train
    else:
        df_seg = df_feat

    cols = [TARGET] + [c for c in df_seg.columns if c != TARGET] if grp_choice.startswith("Semua") else [TARGET, *FEATURE_GROUPS[grp_choice]]
    df_view = df_seg[cols].head(int(max_rows)).copy()
    df_view.index = df_view.index.strftime("%Y-%m-%d %H:%M WIB")

    with f4:
        st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "Unduh Slice CSV",
            data=df_view.to_csv().encode("utf-8"),
            file_name="jakarta_pm25_features_slice.csv",
            mime="text/csv",
            width="stretch",
        )

    st.dataframe(df_view.round(4), width="stretch", height=340)

    col_l, col_r = st.columns([1.25, 0.95], gap="medium")
    with col_l:
        st.subheader("Statistik Deskriptif (Target & 29 Fitur Prediktor)")
        desc = df_seg[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3)
        desc.index.name = "Variabel"
        st.dataframe(desc, width="stretch", height=320)

    with col_r:
        st.subheader("Diagnostik Resolusi Temporal CAMS")
        st.dataframe(cams_diag.round(3), width="stretch")
        st.caption("Rerata |d²| seimbang di jam mod 3 = 0, 1, 2 (5,07–6,45; fraksi datar 5,4–6,2%) → bukan interpolasi linier kaku.")
