"""Tab 1: Visualisasi Prediksi & Evaluasi Sub-Segmen Kritis (Gambar 3.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.pipeline import MODEL_SPECS, RUSH_HOURS, compute_metrics
from src.theme import COLORS, style_plotly


def _extract_overall_ref(leaderboard_df: pd.DataFrame, model_code: str, col: str) -> str:
    row = leaderboard_df.loc[leaderboard_df["Kode"] == model_code, col]
    return str(row.iloc[0]) if not row.empty else "-"


def _render_kpi_card(label: str, val_str: str, delta_html: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val_str}</div>
            {delta_html}
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_error_delta(val: float | None, cmp_val: float | None, cmp_code: str, unit: str = "") -> str:
    if val is None or cmp_val is None or cmp_val == 0:
        return '<div class="kpi-delta-neutral">Pembanding tidak tersedia</div>'
    diff = val - cmp_val
    pct = (diff / cmp_val) * 100.0
    if diff < -1e-6:
        cls, arrow = "kpi-delta-good", "▼"
    elif diff > 1e-6:
        cls, arrow = "kpi-delta-warn", "▲"
    else:
        cls, arrow = "kpi-delta-neutral", "•"
    return f'<div class="{cls}">{arrow} {diff:+.3f}{unit} ({pct:+.2f}%) vs {cmp_code}</div>'


def _format_r2_delta(val: float | None, cmp_val: float | None, cmp_code: str) -> str:
    if val is None or cmp_val is None:
        return '<div class="kpi-delta-neutral">Butuh ≥ 2 titik bervariasi</div>'
    diff = val - cmp_val
    if diff > 1e-6:
        cls, arrow = "kpi-delta-good", "▲"
    elif diff < -1e-6:
        cls, arrow = "kpi-delta-warn", "▼"
    else:
        cls, arrow = "kpi-delta-neutral", "•"
    return f'<div class="{cls}">{arrow} {diff:+.4f} vs {cmp_code}</div>'


def render_prediction_tab(
    df_feat: pd.DataFrame,
    test_df: pd.DataFrame,
    main_code: str,
    cmp_code: str,
    y_pred_main: np.ndarray,
    y_pred_cmp: np.ndarray,
    mask_filtered: pd.Series,
    p95_threshold: float,
    leaderboard_df: pd.DataFrame,
    subseg_df: pd.DataFrame,
) -> None:
    """Render scorecards, interactive actual vs predicted plot, and subsegment/diurnal panels."""
    mask_arr = mask_filtered.to_numpy(dtype=bool)
    n_selected = int(mask_arr.sum())
    if n_selected == 0:
        st.warning(
            "Tidak ada observasi Data Uji yang memenuhi kombinasi rentang tanggal dan filter sub-segmen saat ini. "
            "Silakan perlebar rentang tanggal atau matikan salah satu filter di sidebar."
        )
        return

    y_true_sub = test_df["pm2_5"].to_numpy(dtype=float)[mask_arr]
    y_main_sub = y_pred_main[mask_arr]
    y_cmp_sub = y_pred_cmp[mask_arr]
    idx_sub = test_df.index[mask_arr]

    m_main = compute_metrics(y_true_sub, y_main_sub)
    m_cmp = compute_metrics(y_true_sub, y_cmp_sub)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card(
            "RMSE (Data Uji Terfilter)",
            f"{m_main['rmse']:.3f} µg/m³",
            _format_error_delta(m_main["rmse"], m_cmp["rmse"], cmp_code, " µg/m³"),
            f"10-Seed Overall: {_extract_overall_ref(leaderboard_df, main_code, 'RMSE')}",
        )
    with c2:
        _render_kpi_card(
            "MAE (Data Uji Terfilter)",
            f"{m_main['mae']:.3f} µg/m³",
            _format_error_delta(m_main["mae"], m_cmp["mae"], cmp_code, " µg/m³"),
            f"10-Seed Overall: {_extract_overall_ref(leaderboard_df, main_code, 'MAE')}",
        )
    with c3:
        mape_txt = f"{m_main['mape']:.2f}%" if m_main["mape"] is not None else "N/A"
        _render_kpi_card(
            "MAPE (%)",
            mape_txt,
            _format_error_delta(m_main["mape"], m_cmp["mape"], cmp_code, "%"),
            f"10-Seed Overall: {_extract_overall_ref(leaderboard_df, main_code, 'MAPE (%)')}%",
        )
    with c4:
        r2_txt = f"{m_main['r2']:.3f}" if m_main["r2"] is not None else "N/A"
        _render_kpi_card(
            "Koefisien R²",
            r2_txt,
            _format_r2_delta(m_main["r2"], m_cmp["r2"], cmp_code),
            f"10-Seed Overall: {_extract_overall_ref(leaderboard_df, main_code, 'R2')} · n={n_selected:,}",
        )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # Interactive time series chart
    contiguous = len(idx_sub) > 1 and ((idx_sub[-1] - idx_sub[0]).total_seconds() / 3600.0 + 1.0) <= len(idx_sub) * 1.25
    mode = "lines" if contiguous else "markers"
    fig = go.Figure()

    span_days = (idx_sub[-1] - idx_sub[0]).total_seconds() / 86400.0 if len(idx_sub) > 1 else 0.0
    if span_days <= 21:
        rush_starts = [ts for ts in pd.date_range(idx_sub[0].floor("D"), idx_sub[-1].ceil("D"), freq="1h") if ts.hour in (6, 16)]
        for ts in rush_starts:
            fig.add_vrect(
                x0=ts,
                x1=ts + pd.Timedelta(hours=4),
                fillcolor=COLORS["rush_fill"],
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=idx_sub,
            y=y_true_sub,
            name="Observasi Aktual PM2.5 (CAMS)",
            mode=mode,
            line={"color": COLORS["ink"], "width": 2.2},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx_sub,
            y=y_main_sub,
            name=MODEL_SPECS[main_code].display_name,
            mode=mode,
            line={"color": COLORS["primary"], "width": 2.0},
            marker={"size": 5},
        )
    )
    if cmp_code != main_code:
        fig.add_trace(
            go.Scatter(
                x=idx_sub,
                y=y_cmp_sub,
                name=MODEL_SPECS[cmp_code].display_name,
                mode=mode,
                line={"color": COLORS["secondary"], "width": 1.6, "dash": "dash"},
                marker={"size": 4},
            )
        )

    fig.add_hline(
        y=p95_threshold,
        line_dash="dot",
        line_color=COLORS["danger"],
        line_width=1.5,
        annotation_text=f"Ambang Puncak P95 ({p95_threshold:.1f} µg/m³)",
        annotation_position="top left",
    )
    style_plotly(
        fig,
        title=f"Konsentrasi PM2.5 Aktual vs Prediksi ({idx_sub[0].strftime('%d %b %Y')} – {idx_sub[-1].strftime('%d %b %Y')}, n={n_selected:,} jam)",
        height=430,
    )
    fig.update_xaxes(title_text="Waktu (WIB)")
    fig.update_yaxes(title_text="Konsentrasi PM2.5 (µg/m³)")
    st.plotly_chart(fig, width="stretch")

    # Bottom 2-column critical subsegment + diurnal/seasonal panels
    col_l, col_r = st.columns([1.05, 1.15], gap="medium")
    with col_l:
        st.subheader("Evaluasi Sub-Segmen Kritis (Data Uji)")
        focus_codes = list(dict.fromkeys([main_code, cmp_code, "A", "C", "D'", "F'"]))
        sub_view = subseg_df[subseg_df["Kode"].isin(focus_codes)].copy()
        num_cols = [
            "RMSE total",
            "RMSE puncak (>=P95)",
            "Bias puncak (pred-aktual)",
            "RMSE jam sibuk",
            "RMSE di luar jam sibuk",
            "MAE jam sibuk",
        ]
        st.dataframe(
            sub_view.style.format({c: "{:.3f}" for c in num_cols}),
            width="stretch",
            hide_index=True,
        )
        main_row = subseg_df.loc[subseg_df["Kode"] == main_code].iloc[0]
        raw_row = subseg_df.loc[subseg_df["Kode"] == "C"].iloc[0]
        peak_diff = main_row["RMSE puncak (>=P95)"] - raw_row["RMSE puncak (>=P95)"]
        st.caption(
            f"Ambang P95 Data Uji = {p95_threshold:.1f} µg/m³ (n=198 puncak; n=1.316 jam sibuk). "
            f"Selisih RMSE puncak {main_code} vs Raw (C): {peak_diff:+.3f} µg/m³; "
            f"Bias puncak {main_code}: {main_row['Bias puncak (pred-aktual)']:+.3f} µg/m³."
        )

    with col_r:
        st.subheader("Profil Siklus Diurnal & Galat per Jam")
        chart_opt = st.radio(
            "Pilih tampilan siklus:",
            [
                "Pola Diurnal 24 Jam (PM2.5, Angin & Titik Embun)",
                "Pola Musiman Bulanan (Seluruh Data)",
                "Galat MAE per Jam (Data Uji)",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )
        if chart_opt.startswith("Pola Diurnal"):
            hourly = df_feat.groupby(df_feat.index.hour)[["pm2_5", "wind_speed_10m", "dew_point"]].mean()
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            for h0, h1 in ((6, 9), (16, 19)):
                fig2.add_vrect(x0=h0 - 0.5, x1=h1 + 0.5, fillcolor=COLORS["rush_fill"], line_width=0)
            fig2.add_trace(
                go.Scatter(
                    x=hourly.index,
                    y=hourly["pm2_5"],
                    name="Rata-rata PM2.5 (µg/m³)",
                    mode="lines+markers",
                    line={"color": COLORS["ink"], "width": 2.4},
                ),
                secondary_y=False,
            )
            fig2.add_trace(
                go.Scatter(
                    x=hourly.index,
                    y=hourly["wind_speed_10m"],
                    name="Kecepatan Angin (km/jam)",
                    mode="lines",
                    line={"color": COLORS["accent_blue"], "dash": "dash"},
                ),
                secondary_y=True,
            )
            fig2.add_trace(
                go.Scatter(
                    x=hourly.index,
                    y=hourly["dew_point"],
                    name="Titik Embun (°C)",
                    mode="lines",
                    line={"color": COLORS["primary"], "dash": "dot", "width": 2.2},
                ),
                secondary_y=True,
            )
            style_plotly(fig2, height=340)
            fig2.update_xaxes(title_text="Jam (WIB)", dtick=2)
            fig2.update_yaxes(title_text="PM2.5 (µg/m³)", secondary_y=False)
            fig2.update_yaxes(title_text="Angin (km/jam) / Titik Embun (°C)", secondary_y=True)
            st.plotly_chart(fig2, width="stretch")
        elif chart_opt.startswith("Pola Musiman"):
            monthly = df_feat.groupby(df_feat.index.month)["pm2_5"].mean()
            months_lbl = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
            fig2 = go.Figure(
                go.Bar(
                    x=[months_lbl[m - 1] for m in monthly.index],
                    y=monthly.values,
                    marker_color=COLORS["accent_blue"],
                    text=[f"{v:.1f}" for v in monthly.values],
                    textposition="outside",
                )
            )
            style_plotly(fig2, height=340)
            fig2.update_yaxes(title_text="Rata-rata PM2.5 (µg/m³)", range=[0, max(monthly.values) * 1.15])
            st.plotly_chart(fig2, width="stretch")
        else:
            err_df = pd.DataFrame(
                {
                    "main_err": np.abs(test_df["pm2_5"].to_numpy(dtype=float) - y_pred_main),
                    "cmp_err": np.abs(test_df["pm2_5"].to_numpy(dtype=float) - y_pred_cmp),
                },
                index=test_df.index,
            )
            herr = err_df.groupby(err_df.index.hour).mean()
            fig2 = go.Figure()
            for h0, h1 in ((6, 9), (16, 19)):
                fig2.add_vrect(x0=h0 - 0.5, x1=h1 + 0.5, fillcolor=COLORS["rush_fill"], line_width=0)
            fig2.add_trace(
                go.Scatter(
                    x=herr.index,
                    y=herr["main_err"],
                    name=f"Model {main_code}",
                    mode="lines+markers",
                    line={"color": COLORS["primary"], "width": 2.2},
                )
            )
            if cmp_code != main_code:
                fig2.add_trace(
                    go.Scatter(
                        x=herr.index,
                        y=herr["cmp_err"],
                        name=f"Model {cmp_code}",
                        mode="lines+markers",
                        line={"color": COLORS["secondary"], "width": 1.8, "dash": "dash"},
                    )
                )
            style_plotly(fig2, height=340)
            fig2.update_xaxes(title_text="Jam (WIB)", dtick=2)
            fig2.update_yaxes(title_text="MAE per Jam (µg/m³)")
            st.plotly_chart(fig2, width="stretch")
