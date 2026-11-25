"""Centralized styling, semantic palette, and Plotly layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "ink": "#0F172A",
    "muted": "#475569",
    "surface": "#FFFFFF",
    "bg": "#F8FAFC",
    "border": "#E2E8F0",
    "primary": "#0F766E",      # Teal - proposed / primary evaluation model
    "secondary": "#D97706",    # Amber - comparison / baseline model
    "accent_blue": "#2563EB",  # Blue - physics / climate series
    "danger": "#DC2626",       # Crimson - P95 extreme threshold
    "rush_fill": "rgba(250, 204, 21, 0.16)",
    "rush_line": "rgba(202, 138, 4, 0.45)",
}

CATEGORY_COLORS = {
    "Lag Autoregresif": "#DC2626",
    "Siklik Temporal": "#16A34A",
    "Fisika Iklim": "#2563EB",
    "Cuaca Mentah": "#64748B",
}


def inject_custom_css() -> None:
    """Inject clean scientific dashboard CSS into the active Streamlit page."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .header-banner {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #0F766E;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .header-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.25rem 0;
        }
        .header-sub {
            font-size: 0.9rem;
            color: #475569;
            margin: 0;
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            height: 100%;
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #64748B;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .kpi-delta-good {
            font-size: 0.8rem;
            font-weight: 600;
            color: #0F766E;
            margin-top: 0.25rem;
        }
        .kpi-delta-warn {
            font-size: 0.8rem;
            font-weight: 600;
            color: #B45309;
            margin-top: 0.25rem;
        }
        .kpi-delta-neutral {
            font-size: 0.8rem;
            font-weight: 500;
            color: #475569;
            margin-top: 0.25rem;
        }
        .kpi-foot {
            font-size: 0.75rem;
            color: #64748B;
            margin-top: 0.2rem;
        }
        .info-panel {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            font-size: 0.85rem;
            color: #334155;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #FFFFFF;
            padding: 0.35rem;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 7px;
            padding: 0.45rem 1rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent accessible Plotly styling across all figures."""
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": COLORS["ink"]}, "x": 0.01, "y": 0.96},
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "sans-serif", "color": COLORS["ink"], "size": 12},
        margin={"l": 50, "r": 24, "t": 74 if title else 42, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig
