"""Tab 3: Leaderboard Studi Ablasi (11 Model, 10 Seeds) & Explainable AI (Gambar 3.8)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.theme import CATEGORY_COLORS, COLORS, style_plotly


def _parse_mean_std(text: str) -> tuple[float, float]:
    s = str(text).strip()
    if "±" in s:
        parts = s.split("±")
        return float(parts[0].strip()), float(parts[1].strip())
    return float(s), 0.0


def render_leaderboard_tab(
    leaderboard_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    tuning_summary: dict,
) -> None:
    """Render 11-model ablation leaderboard, Top-15 XAI chart, and tuning configuration summary."""
    st.subheader("Papan Peringkat (Leaderboard) 11 Model Studi Ablasi — Data Uji (10 Random Seeds)")

    rmse_vals, rmse_errs = zip(*[_parse_mean_std(v) for v in leaderboard_df["RMSE"]])
    mae_vals, mae_errs = zip(*[_parse_mean_std(v) for v in leaderboard_df["MAE"]])
    codes = leaderboard_df["Kode"].tolist()
    short_lbl = [
        f"{r['Kode']} ({r['Model'].split('(')[-1].split(')')[0][:12]})" if "(" in r["Model"] else r["Kode"]
        for _, r in leaderboard_df.iterrows()
    ]
    bar_colors = [
        COLORS["primary"] if c == "F'" else (COLORS["accent_blue"] if c == "D'" else ("#94A3B8" if c in ("A", "B", "B2") else "#64748B"))
        for c in codes
    ]

    fig_lb = make_subplots(rows=1, cols=2, subplot_titles=("RMSE Data Uji (mean ± std)", "MAE Data Uji (mean ± std)"))
    fig_lb.add_trace(
        go.Bar(
            x=codes,
            y=rmse_vals,
            error_y={"type": "data", "array": rmse_errs, "visible": True},
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in rmse_vals],
            textposition="outside",
            name="RMSE",
            customdata=short_lbl,
            hovertemplate="%{x}: %{y:.3f} µg/m³<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_lb.add_trace(
        go.Bar(
            x=codes,
            y=mae_vals,
            error_y={"type": "data", "array": mae_errs, "visible": True},
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in mae_vals],
            textposition="outside",
            name="MAE",
            hovertemplate="%{x}: %{y:.3f} µg/m³<extra></extra>",
        ),
        row=1,
        col=2,
    )
    style_plotly(fig_lb, height=340)
    fig_lb.update_layout(showlegend=False)
    fig_lb.update_yaxes(title_text="RMSE (µg/m³)", range=[0, max(rmse_vals) * 1.18], row=1, col=1)
    fig_lb.update_yaxes(title_text="MAE (µg/m³)", range=[0, max(mae_vals) * 1.18], row=1, col=2)
    st.plotly_chart(fig_lb, width="stretch")

    st.dataframe(leaderboard_df, width="stretch", hide_index=True)

    st.markdown("<div style='height: 0.65rem'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.2, 0.95], gap="medium")

    with col_l:
        st.subheader("Top 15 Fitur Terpenting (Model G, Data Uji)")
        imp_mode = st.radio(
            "Metrik XAI:",
            ["Permutation Importance (Kenaikan RMSE µg/m³)", "Gain Importance (XGBoost Tree Split)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if imp_mode.startswith("Permutation"):
            col_name = "PermImp (naik RMSE)"
            x_title = "Kenaikan RMSE saat fitur diacak (µg/m³)"
            show_err = True
        else:
            col_name = "Gain Importance"
            x_title = "Proporsi Gain Importance"
            show_err = False

        top15 = fi_df.sort_values(col_name, ascending=False).head(15).iloc[::-1]
        fig_fi = go.Figure()
        for cat_name, cat_color in CATEGORY_COLORS.items():
            sub = top15[top15["Kategori"] == cat_name]
            if sub.empty:
                continue
            fig_fi.add_trace(
                go.Bar(
                    y=sub["Fitur"],
                    x=sub[col_name],
                    orientation="h",
                    name=cat_name,
                    marker_color=cat_color,
                    error_x={"type": "data", "array": sub["Std"], "visible": True} if show_err else None,
                    text=[f"{v:.3f}" for v in sub[col_name]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
        style_plotly(fig_fi, height=450)
        fig_fi.update_layout(barmode="overlay", yaxis={"categoryorder": "array", "categoryarray": top15["Fitur"].tolist()})
        fig_fi.update_xaxes(title_text=x_title, range=[0, float(top15[col_name].max()) * 1.24])
        st.plotly_chart(fig_fi, width="stretch")

        cat_tbl = (
            fi_df.groupby("Kategori")[["PermImp (naik RMSE)", "Gain Importance"]]
            .sum()
            .sort_values("PermImp (naik RMSE)", ascending=False)
            .round(3)
        )
        st.markdown("**Total Kontribusi per Kategori Fitur (Permutation vs Gain)**")
        st.dataframe(cat_tbl, width="stretch")

    with col_r:
        st.subheader("Ringkasan Konfigurasi Optimal Hasil Tuning")
        best_cfg = tuning_summary["best_cfg"]
        cfg_rows = [
            {
                "Skenario": f"{k}' ({'Full / Model G' if k=='F' else ('Raw' if k=='C' else ('+Temporal' if k=='D' else '+Iklim'))})",
                "n_estimators": v["n_estimators"],
                "max_depth": v["max_depth"],
                "lr": v["learning_rate"],
                "subsample": v["subsample"],
                "colsample": v["colsample_bytree"],
                "Val RMSE": tuning_summary["top5"][k][0]["Val RMSE"],
            }
            for k, v in best_cfg.items()
        ]
        st.dataframe(pd.DataFrame(cfg_rows), width="stretch", hide_index=True)

        st.markdown("**Top 5 Kombinasi ParameterGrid — Set 4 Full (Data Validasi)**")
        top5_full = pd.DataFrame(tuning_summary["top5"]["F"])
        top5_full.insert(0, "Peringkat", range(1, len(top5_full) + 1))
        st.dataframe(top5_full.round(3), width="stretch", hide_index=True)
        st.caption(
            f"Grid Search {tuning_summary['grid_count']} kombinasi/set · Early Stopping {tuning_summary['early_stopping_rounds']} ronde · "
            f"{len(tuning_summary['seeds'])} Random Seeds."
        )
