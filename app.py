# ══════════════════════════════════════════════════════════════
# CreciPSI v6.0 — Diseño Fintech/Data + Formulario estructurado
# FMVZ-UNAM | Diplomado IA en Salud Global 2025-2026
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import warnings
import requests
import os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CreciPSI",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0a0a;
    color: #e8e8e8;
}

#MainMenu, footer, .stDeployButton, .stDecoration { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* ── Layout principal ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Topbar ── */
.topbar {
    background: #0d0d0d;
    border-bottom: 1px solid #1e1e1e;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade8088;
}
.topbar-logo .name {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.3px;
}
.topbar-logo .sub {
    font-size: 11px;
    color: #555;
    margin-left: 2px;
}
.topbar-badge {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    color: #666;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Métricas strip ── */
.metrics-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
}
.metric-cell {
    background: #0d0d0d;
    padding: 14px 20px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.metric-cell .mv {
    font-size: 20px;
    font-weight: 600;
    color: #f0f0f0;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-cell .ml {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d0d !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding: 0 24px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #555 !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #4ade80 !important;
    border-bottom-color: #4ade80 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: #0a0a0a !important;
    padding: 24px !important;
}

/* ── Cards ── */
.card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 16px 18px;
}
.card-dark {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 16px 18px;
}
.section-label {
    font-size: 10px;
    font-weight: 600;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-label::before {
    content: '';
    width: 3px; height: 3px;
    border-radius: 50%;
    background: #4ade80;
}

/* ── Patron chips ── */
.patron-normal    { background: #0f2a1a; color: #4ade80; border: 1px solid #1a4a2a; }
.patron-superior  { background: #0f1a3a; color: #60a5fa; border: 1px solid #1a2a5a; }
.patron-inferior  { background: #2a1a0f; color: #fb923c; border: 1px solid #4a2a1a; }
.patron-irregular { background: #2a0f0f; color: #f87171; border: 1px solid #4a1a1a; }
.patron-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 12px 0;
}
.stat-item {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.stat-item .sv {
    font-size: 18px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: #f0f0f0;
    line-height: 1;
}
.stat-item .sl {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-top: 3px;
}
.stat-item.green .sv { color: #4ade80; }
.stat-item.amber .sv { color: #fb923c; }
.stat-item.red .sv   { color: #f87171; }
.stat-item.blue .sv  { color: #60a5fa; }

/* ── Indicaciones ── */
.ind-box {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 10px;
}
.ind-lbl {
    font-size: 10px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 8px;
}
.ind-item {
    display: flex;
    gap: 8px;
    font-size: 12px;
    color: #aaa;
    margin-bottom: 5px;
    align-items: flex-start;
    line-height: 1.5;
}
.ind-item:last-child { margin-bottom: 0; }
.ind-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #444;
    flex-shrink: 0;
    margin-top: 6px;
}
.ind-green .ind-dot { background: #4ade80; }
.ind-green { color: #86efac; }
.ind-amber .ind-dot { background: #fb923c; }
.ind-amber { color: #fdba74; }
.ind-red .ind-dot { background: #f87171; }
.ind-red { color: #fca5a5; }
.ind-blue .ind-dot { background: #60a5fa; }
.ind-blue { color: #93c5fd; }

/* ── Alerts ── */
.alert-ok   { background: #0f2a1a; border: 1px solid #1a4a2a; border-radius: 7px; padding: 8px 12px; font-size: 12px; color: #86efac; margin: 6px 0; }
.alert-warn { background: #2a1a0f; border: 1px solid #4a2a1a; border-radius: 7px; padding: 8px 12px; font-size: 12px; color: #fdba74; margin: 6px 0; }

/* ── Inputs de Streamlit ── */
.stNumberInput > div > div > input {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    text-align: center !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px #4ade8022 !important;
}

.stTextInput > div > div > input {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px #4ade8022 !important;
}

.stTextArea > div > div > textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

.stSelectbox > div > div {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
}

/* ── Radio buttons ── */
.stRadio > div {
    flex-direction: row !important;
    gap: 8px !important;
}
.stRadio > div > label {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    padding: 5px 12px !important;
    font-size: 12px !important;
    color: #888 !important;
    cursor: pointer !important;
}
.stRadio > div > label:has(input:checked) {
    border-color: #4ade80 !important;
    color: #4ade80 !important;
    background: #0f2a1a !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: #2a2a2a !important;
}
.stSlider > div > div > div > div {
    background: #4ade80 !important;
}

/* ── Botones ── */
.stButton > button {
    background: #4ade80 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 7px !important;
    color: #888 !important;
    font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-top: none !important;
}

/* ── Dataframes ── */
.stDataFrame {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
}

/* ── Labels ── */
.stTextInput label, .stNumberInput label,
.stTextArea label, .stSelectbox label,
.stRadio label, .stSlider label {
    color: #666 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: #1e1e1e;
    margin: 16px 0;
}

/* ── Month grid ── */
.month-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.month-card {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 10px 12px;
}
.month-card.filled {
    border-color: #1a4a2a;
    background: #0a1a0f;
}
.month-card .mc-label {
    font-size: 10px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.month-card.filled .mc-label { color: #4ade80; }

/* ── Matplotlib style ── */
</style>
""", unsafe_allow_html=True)


# ── CONFIG MATPLOTLIB ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111111",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#2a2a2a",
    "axes.labelcolor":   "#888888",
    "axes.titlecolor":   "#cccccc",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#1e1e1e",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  "#111111",
    "legend.edgecolor":  "#2a2a2a",
    "legend.labelcolor": "#aaaaaa",
    "legend.fontsize":   9,
    "text.color":        "#cccccc",
    "font.family":       "sans-serif",
})

VERDE  = "#4ade80"
AZUL   = "#60a5fa"
AMBER  = "#fb923c"
ROJO   = "#f87171"
GRIS   = "#555555"
C = {"M": VERDE, "H": AZUL}


# ── CARGAR MODELOS ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar():
    with open("stats_ref_final.pkl",    "rb") as f: sr = pickle.load(f)
    with open("stats_alzada_final.pkl", "rb") as f: sa = pickle.load(f)
    with open("modelo_peso_v2.pkl",     "rb") as f: mp = pickle.load(f)
    with open("modelo_alzada.pkl",      "rb") as f: ma = pickle.load(f)
    return sr, sa, mp, ma

try:
    stats_ref, stats_alz, mod_peso, mod_alz = cargar()
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()


# ── TOPBAR ───────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">
    <div class="dot"></div>
    <span class="name">CreciPSI</span>
    <span class="sub">Monitor inteligente de crecimiento equino</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="topbar-badge">FMVZ-UNAM</span>
    <span class="topbar-badge">DIASG 2025-2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MÉTRICAS STRIP ───────────────────────────────────────────
st.markdown("""
<div class="metrics-strip">
  <div class="metric-cell"><div class="mv">217</div><div class="ml">Potros PSI</div></div>
  <div class="metric-cell"><div class="mv">4,175</div><div class="ml">Mediciones</div></div>
  <div class="metric-cell"><div class="mv">10 años</div><div class="ml">2015 – 2025</div></div>
  <div class="metric-cell"><div class="mv">0.964</div><div class="ml">R² modelo</div></div>
  <div class="metric-cell"><div class="mv">±15 kg</div><div class="ml">Error medio</div></div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Curvas de referencia",
    "Evaluar potro",
    "Predictor",
    "Comparación internacional",
    "Reporte IA",
    "Metodología",
])


# ══════════════════════════════════════════════════════════════
# HELPER: graficar curvas estilo dark
# ══════════════════════════════════════════════════════════════

def fig_curvas(stats, color, titulo, ylabel, ylim,
               meses_anot, fmt_anot, offset_anot,
               datos_potro=None, nombre_potro=None,
               punto_pred=None, edad_pred=None,
               ancho=11, alto=4.5):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    edades = stats["edad_meses"]

    ax.fill_between(edades, stats.p10, stats.p90,
                    alpha=0.06, color=color)
    ax.fill_between(edades, stats.p25, stats.p75,
                    alpha=0.18, color=color,
                    label="Rango normal P25–P75")
    ax.plot(edades, stats.p50, color=color,
            linewidth=2, label="Mediana P50", zorder=3)
    ax.plot(edades, stats.p10, color=color,
            linewidth=0.7, linestyle=":", alpha=0.35)
    ax.plot(edades, stats.p90, color=color,
            linewidth=0.7, linestyle=":", alpha=0.35)

    for mes in meses_anot:
        f = stats[stats.edad_meses == mes]
        if len(f) == 0:
            continue
        v = f["p50"].values[0]
        ax.annotate(fmt_anot.format(v),
                    xy=(mes, v), xytext=(mes + 0.6, v + offset_anot),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="#111111", ec=color, alpha=0.9))

    if datos_potro and len(datos_potro) >= 2:
        eds = sorted(datos_potro.keys())
        vls = [datos_potro[e] for e in eds]
        ax.plot(eds, vls, color=AMBER, linewidth=2.2,
                marker="o", markersize=6,
                label=nombre_potro or "Potro evaluado", zorder=5)
        for e, v in zip(eds, vls):
            ax.annotate(f"{v:.0f}" if offset_anot > 5 else f"{v:.2f}",
                        xy=(e, v),
                        xytext=(e + 0.3, v + offset_anot * 0.55),
                        fontsize=7.5, color=AMBER,
                        arrowprops=dict(arrowstyle="-",
                                        color=AMBER, lw=0.7, alpha=0.5))

    if punto_pred is not None and edad_pred is not None:
        ax.axvline(x=edad_pred, color="#333", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.scatter([edad_pred], [punto_pred],
                   color=AMBER, s=160, zorder=7,
                   edgecolors="#111", linewidths=2,
                   label=f"Pred: {punto_pred:.0f}" if offset_anot > 5
                         else f"Pred: {punto_pred:.3f}")
        ax.annotate(f"  {punto_pred:.0f} kg" if offset_anot > 5
                    else f"  {punto_pred:.3f} m",
                    xy=(edad_pred, punto_pred),
                    xytext=(edad_pred + 0.8, punto_pred + offset_anot * 1.1),
                    fontsize=9, color=AMBER, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#1a0f00", ec=AMBER, alpha=0.95))

    ax.set_xlabel(ylabel.split("(")[0].strip() + " — edad (meses)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=10, pad=10)
    ax.legend(loc="upper left", framealpha=0.8)
    xlim_min = -0.3 if 0 in (edades.values if hasattr(edades, "values") else edades) else 0.5
    ax.set_xlim(xlim_min, 22.5)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 1 — CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-label">Curvas percentiladas del rancho</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        sx1 = st.radio("Sexo", ["Machos", "Hembras"], key="sx1")
    with c2:
        vr1 = st.radio("Variable", ["Peso (kg)", "Alzada (m)"], key="vr1")

    sk1 = "M" if sx1 == "Machos" else "H"
    n1  = 111 if sk1 == "M" else 106
    col1 = C[sk1]

    if "Peso" in vr1:
        fig1 = fig_curvas(
            stats_ref[f"stats_{sk1}"], col1,
            f"Peso corporal — {'Machos' if sk1=='M' else 'Hembras'} (n={n1})",
            "Peso (kg)", (20, 570), [0, 6, 12, 18], "{:.0f} kg", 20
        )
    else:
        fig1 = fig_curvas(
            stats_alz[f"stats_{sk1}"], col1,
            f"Alzada a la cruz — {'Machos' if sk1=='M' else 'Hembras'} (n={n1})",
            "Alzada (m)", (0.85, 1.68), [6, 12, 18], "{:.2f} m", 0.012
        )

    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("Ver tabla de valores de referencia"):
        st_d1 = (stats_ref[f"stats_{sk1}"] if "Peso" in vr1
                 else stats_alz[f"stats_{sk1}"])
        t1 = st_d1[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
        t1.columns = ["Edad","P10","P25","P50","P75","P90","N"]
        st.dataframe(t1.round(1 if "Peso" in vr1 else 3),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="alert-ok" style="margin-top:12px">
        Rango normal (P25–P75): zona donde se encuentra el 50% central de la población.
        Valores por debajo del P10 o por encima del P90 justifican evaluación clínica adicional.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUAR POTRO
# ══════════════════════════════════════════════════════════════

with tab2:
    MESES_CLAVE = [1, 3, 6, 9, 12, 18]
    MESES_EXTRA = [2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22]

    # ── Layout: formulario izq, resultado der ──
    form_col, result_col = st.columns([1, 1.4], gap="large")

    # ── FORMULARIO ──────────────────────────────────────────
    with form_col:
        st.markdown('<div class="section-label">Datos del paciente</div>',
                    unsafe_allow_html=True)

        nombre2 = st.text_input("Identificador",
                                placeholder="Ej. Hijo de Mila Race",
                                key="nombre2")

        c_sx, c_nac = st.columns(2)
        with c_sx:
            sx2 = st.radio("Sexo", ["Macho", "Hembra"], key="sx2")
        with c_nac:
            pnac2 = st.number_input("Peso al nacer (kg)",
                                    min_value=0.0, max_value=80.0,
                                    value=0.0, step=0.5, key="pnac2")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Mediciones mensuales</div>',
                    unsafe_allow_html=True)
        st.caption("Deja en 0 los meses sin medición. Mínimo 2 meses con peso.")

        pesos2   = {}
        alzadas2 = {}
        if pnac2 > 0:
            pesos2[0] = pnac2

        cols_k = st.columns(2)
        for i, mes in enumerate(MESES_CLAVE):
            with cols_k[i % 2]:
                st.markdown(f"<div style='font-size:11px;color:#4ade80;font-weight:600;"
                            f"margin:8px 0 4px;text-transform:uppercase;letter-spacing:0.4px'>"
                            f"Mes {mes}</div>", unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    pv = st.number_input("kg", min_value=0.0, max_value=700.0,
                                         value=0.0, step=1.0, key=f"p2_{mes}")
                with cb:
                    av = st.number_input("m", min_value=0.0, max_value=2.0,
                                         value=0.0, step=0.01, key=f"a2_{mes}")
                if pv > 0: pesos2[mes] = pv
                if av > 0: alzadas2[mes] = av

        with st.expander("+ Agregar meses adicionales"):
            cols_e = st.columns(3)
            for i, mes in enumerate(MESES_EXTRA):
                with cols_e[i % 3]:
                    st.markdown(f"<div style='font-size:10px;color:#555;"
                                f"margin:6px 0 3px;text-transform:uppercase'>"
                                f"Mes {mes}</div>", unsafe_allow_html=True)
                    ca2, cb2 = st.columns(2)
                    with ca2:
                        pv2 = st.number_input("kg", min_value=0.0, max_value=700.0,
                                              value=0.0, step=1.0, key=f"p2_{mes}")
                    with cb2:
                        av2 = st.number_input("m", min_value=0.0, max_value=2.0,
                                              value=0.0, step=0.01, key=f"a2_{mes}")
                    if pv2 > 0: pesos2[mes] = pv2
                    if av2 > 0: alzadas2[mes] = av2

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        analizar2 = st.button("Analizar crecimiento →",
                              type="primary", key="btn2")

    # ── RESULTADO ───────────────────────────────────────────
    with result_col:
        if not analizar2:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        text-align:center;padding:40px 20px">
                <div style="width:48px;height:48px;border:1px solid #1e1e1e;
                            border-radius:12px;display:flex;align-items:center;
                            justify-content:center;margin:0 auto 16px">
                    <span style="font-size:22px">🐴</span>
                </div>
                <div style="font-size:14px;font-weight:500;color:#555;margin-bottom:6px">
                    Ingresa los datos y presiona analizar
                </div>
                <div style="font-size:12px;color:#333;line-height:1.6">
                    El sistema calculará los percentiles, clasificará el patrón
                    de crecimiento y generará indicaciones clínicas específicas.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            meses_p = sorted([m for m in pesos2 if pesos2[m] > 0 and m > 0])
            if len(meses_p) < 2:
                st.warning("Ingresa al menos 2 mediciones de peso (meses > 0).")
            else:
                sk2 = "M" if sx2 == "Macho" else "H"
                sp2 = stats_ref[f"stats_{sk2}"]
                sa2 = stats_alz[f"stats_{sk2}"]
                nombre_d = nombre2 or "Potro evaluado"

                filas2 = []
                alertas2 = []
                alto_cnt = bajo_cnt = 0

                for mes in meses_p:
                    ref = sp2[sp2.edad_meses == mes]
                    if ref.empty: continue
                    peso = pesos2[mes]
                    p10=ref["p10"].values[0]; p25=ref["p25"].values[0]
                    p50=ref["p50"].values[0]; p75=ref["p75"].values[0]
                    p90=ref["p90"].values[0]
                    diff = ((peso - p50) / p50) * 100

                    if peso < p10:    zona="MUY BAJO"; alerta=True;  bajo_cnt+=1
                    elif peso < p25:  zona="BAJO";     alerta=True;  bajo_cnt+=1
                    elif peso <= p75: zona="NORMAL";   alerta=False
                    elif peso <= p90: zona="ALTO";     alerta=False; alto_cnt+=1
                    else:             zona="MUY ALTO"; alerta=True;  alto_cnt+=1

                    if alerta: alertas2.append(mes)

                    alz_zona = None
                    if mes in alzadas2 and alzadas2[mes] > 0:
                        rfa = sa2[sa2.edad_meses == mes]
                        if not rfa.empty:
                            if alzadas2[mes] < rfa["p25"].values[0]:   alz_zona="Baja"
                            elif alzadas2[mes] <= rfa["p75"].values[0]:alz_zona="Normal"
                            else:                                       alz_zona="Alta"

                    filas2.append({
                        "mes":mes,"peso":peso,"p10":p10,"p25":p25,
                        "p50":p50,"p75":p75,"p90":p90,"diff":diff,
                        "zona":zona,"alerta":alerta,
                        "alzada":alzadas2.get(mes,None),"alz_zona":alz_zona
                    })

                n_f = len(filas2)
                pct_alto = alto_cnt/n_f if n_f>0 else 0
                pct_bajo = bajo_cnt/n_f if n_f>0 else 0
                vals_p   = [f["peso"] for f in filas2]
                perdidas = sum(1 for i in range(1,len(vals_p)) if vals_p[i]<vals_p[i-1])
                caida    = any((vals_p[i]-vals_p[i-1])/vals_p[i-1]*100<-8
                               for i in range(1,len(vals_p)))

                if (perdidas>=4) or caida:
                    patron2="Patrón Irregular"; cls_p="patron-irregular"
                    inds=[("red","Evaluación clínica urgente — pérdida de peso detectada."),
                          ("red","Descartar parasitosis, enfermedad GI o estrés severo."),
                          ("amber","Revisar calidad y cantidad del alimento."),
                          ("amber","Verificar acceso a agua limpia y comedero.")]
                elif pct_alto>=0.60:
                    patron2="Patrón Superior"; cls_p="patron-superior"
                    inds=[("blue","Crecimiento excelente — por encima del P75 en la mayoría de meses."),
                          ("","Mantener el plan nutricional y de manejo actual."),
                          ("amber","Vigilar condición corporal para evitar sobrepeso tardío.")]
                elif pct_bajo>=0.60:
                    patron2="Patrón Inferior"; cls_p="patron-inferior"
                    inds=[("amber","Revisar aporte energético — incrementar concentrado."),
                          ("amber","Evaluar desparasitación — alta carga reduce absorción."),
                          ("amber","Verificar forraje y acceso a agua limpia."),
                          ("","Repetir evaluación en 4 semanas tras ajuste.")]
                else:
                    patron2="Patrón Normal"; cls_p="patron-normal"
                    inds=[("green","Mantener el programa de manejo y alimentación actual."),
                          ("green","Continuar con pesajes mensuales para seguimiento.")]
                    if alertas2:
                        inds.append(("amber",f"Vigilar meses con alerta: {alertas2}."))

                # Chip de patrón
                st.markdown(
                    f'<div class="patron-chip {cls_p}" style="margin-bottom:12px">'
                    f'{patron2}</div>',
                    unsafe_allow_html=True
                )

                # Stats
                ganancia = round(vals_p[-1]-vals_p[0]) if len(vals_p)>=2 else 0
                norm_pct = round(sum(1 for f in filas2 if f["zona"]=="NORMAL")/n_f*100)
                cls_n = "green" if norm_pct>=60 else ("amber" if norm_pct>=40 else "red")
                cls_a = "green" if len(alertas2)==0 else ("amber" if len(alertas2)<=2 else "red")

                st.markdown(f"""
                <div class="stat-grid">
                  <div class="stat-item {cls_n}">
                    <div class="sv">{norm_pct}%</div>
                    <div class="sl">En rango</div>
                  </div>
                  <div class="stat-item">
                    <div class="sv">+{ganancia}</div>
                    <div class="sl">kg ganados</div>
                  </div>
                  <div class="stat-item {cls_a}">
                    <div class="sv">{len(alertas2)}</div>
                    <div class="sl">Alertas</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if len(alertas2)==0:
                    st.markdown('<div class="alert-ok">Sin alertas — crecimiento dentro del rango en todos los meses evaluados</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-warn">Alertas en meses: {alertas2}</div>',
                                unsafe_allow_html=True)

                # Indicaciones
                ind_html = "".join([
                    f'<div class="ind-item ind-{c if c else ""}">'
                    f'<div class="ind-dot"></div><span>{t}</span></div>'
                    for c, t in inds
                ])
                st.markdown(
                    f'<div class="ind-box"><div class="ind-lbl">Indicaciones clínicas</div>'
                    f'{ind_html}</div>',
                    unsafe_allow_html=True
                )

                # Gráficas
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                pesos_dict = {f["mes"]:f["peso"] for f in filas2}
                ylim_p = (50, max(max(pesos_dict.values())+60, 520))
                fig_p = fig_curvas(
                    sp2, C[sk2],
                    f"Peso — {nombre_d} vs. curvas de referencia",
                    "Peso (kg)", ylim_p,
                    [6,12,18], "{:.0f} kg", 20,
                    datos_potro=pesos_dict, nombre_potro=nombre_d,
                    ancho=9, alto=4
                )
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)

                alz_dict = {m:alzadas2[m] for m in alzadas2 if alzadas2[m]>0 and m>0}
                if len(alz_dict)>=2:
                    fig_a = fig_curvas(
                        sa2, C[sk2],
                        f"Alzada — {nombre_d} vs. curvas de referencia",
                        "Alzada (m)", (0.85, 1.70),
                        [6,12,18], "{:.2f} m", 0.012,
                        datos_potro=alz_dict, nombre_potro=nombre_d,
                        ancho=9, alto=3.5
                    )
                    st.pyplot(fig_a, use_container_width=True)
                    plt.close(fig_a)

                with st.expander("Ver tabla detallada mes a mes"):
                    df2 = pd.DataFrame([{
                        "Mes":f["mes"],"Peso(kg)":f["peso"],
                        "P25":round(f["p25"],1),"P50":round(f["p50"],1),
                        "P75":round(f["p75"],1),
                        "vs P50":f'{f["diff"]:+.1f}%',
                        "Estado":f["zona"],
                        "Alzada(m)":f["alzada"] if f["alzada"] else "—",
                        "Estado alz":f["alz_zona"] if f["alz_zona"] else "—",
                    } for f in filas2])

                    def color_e(v):
                        if "BAJO" in str(v):  return "color:#fdba74"
                        elif "ALTO" in str(v): return "color:#93c5fd"
                        elif "NORMAL" in str(v):return "color:#86efac"
                        return ""

                    try:
                        styled = df2.style.map(color_e, subset=["Estado"])
                    except AttributeError:
                        styled = df2.style.applymap(color_e, subset=["Estado"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR
# ══════════════════════════════════════════════════════════════

with tab3:
    form_p, result_p = st.columns([1, 1.4], gap="large")

    with form_p:
        st.markdown('<div class="section-label">Parámetros</div>',
                    unsafe_allow_html=True)
        sx3 = st.radio("Sexo", ["Macho", "Hembra"], key="sx3")
        edad3 = st.slider("Edad (meses)", 1, 22, 6, key="edad3")
        alz3  = st.number_input("Alzada actual (m) — opcional",
                                min_value=0.0, max_value=2.0,
                                value=0.0, step=0.01, key="alz3",
                                help="Con alzada: R²=0.9641 · Sin alzada: R²=0.9458")
        if alz3 > 0:
            st.markdown('<div class="alert-ok">Modelo mejorado activo — R²=0.9641</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#111;border:1px solid #1e1e1e;'
                        'border-radius:7px;padding:8px 12px;font-size:12px;color:#555;'
                        'margin:6px 0">Ingresa la alzada para mayor precisión</div>',
                        unsafe_allow_html=True)

    with result_p:
        sk3   = "M" if sx3=="Macho" else "H"
        sbin3 = 1 if sk3=="M" else 0
        sp3   = stats_ref[f"stats_{sk3}"]
        sa3   = stats_alz[f"stats_{sk3}"]
        ref_a3= sa3[sa3.edad_meses==edad3]
        alz_m3= ref_a3["p50"].values[0] if len(ref_a3)>0 else 1.35
        alz_u3= alz3 if alz3>0 else alz_m3
        peso3 = mod_peso.predict([[sbin3, edad3, alz_u3]])[0]
        alzp3 = mod_alz.predict([[sbin3, edad3, peso3]])[0]
        ref_p3= sp3[sp3.edad_meses==edad3]

        if len(ref_p3)>0:
            p25r=ref_p3["p25"].values[0]
            p50r=ref_p3["p50"].values[0]
            p75r=ref_p3["p75"].values[0]
            if peso3<p25r:    pos="Inferior"; col_pos=AMBER; bg_pos="#1a0f00"
            elif peso3<=p75r: pos="Normal";   col_pos=VERDE; bg_pos="#0a1a0f"
            else:             pos="Superior"; col_pos=AZUL;  bg_pos="#0a0f1a"

            st.markdown(f"""
            <div style="background:{bg_pos};border:1px solid {col_pos}33;
                        border-radius:10px;padding:16px 20px;margin-bottom:12px;
                        text-align:center">
                <div style="font-size:11px;color:{col_pos}88;text-transform:uppercase;
                            letter-spacing:0.6px;margin-bottom:4px">Peso predicho · mes {edad3}</div>
                <div style="font-size:36px;font-weight:700;color:{col_pos};
                            font-family:'JetBrains Mono',monospace;line-height:1">{peso3:.0f} kg</div>
                <div style="font-size:11px;color:{col_pos}88;margin-top:4px">{pos} al rango normal</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px">
                <div class="stat-item">
                    <div class="sv">{p25r:.0f}</div>
                    <div class="sl">P25 rancho</div>
                </div>
                <div class="stat-item blue">
                    <div class="sv">{p50r:.0f}</div>
                    <div class="sl">P50 rancho</div>
                </div>
                <div class="stat-item">
                    <div class="sv">{p75r:.0f}</div>
                    <div class="sl">P75 rancho</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(ref_a3)>0:
                a25=ref_a3["p25"].values[0]; a50=ref_a3["p50"].values[0]
                a75=ref_a3["p75"].values[0]
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px">
                    <div class="stat-item green">
                        <div class="sv">{alzp3:.3f}</div>
                        <div class="sl">Alzada pred.</div>
                    </div>
                    <div class="stat-item">
                        <div class="sv">{a25:.3f}</div>
                        <div class="sl">P25</div>
                    </div>
                    <div class="stat-item blue">
                        <div class="sv">{a50:.3f}</div>
                        <div class="sl">P50</div>
                    </div>
                    <div class="stat-item">
                        <div class="sv">{a75:.3f}</div>
                        <div class="sl">P75</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Gráfica predictor
        edades_g=list(range(1,23))
        saM3=stats_alz["stats_M"]; saH3=stats_alz["stats_H"]
        spM3=stats_ref["stats_M"]; spH3=stats_ref["stats_H"]
        aM3=[saM3[saM3.edad_meses==e]["p50"].values[0]
             if len(saM3[saM3.edad_meses==e])>0 else 1.35 for e in edades_g]
        aH3=[saH3[saH3.edad_meses==e]["p50"].values[0]
             if len(saH3[saH3.edad_meses==e])>0 else 1.33 for e in edades_g]
        pM3=[mod_peso.predict([[1,e,a]])[0] for e,a in zip(edades_g,aM3)]
        pH3=[mod_peso.predict([[0,e,a]])[0] for e,a in zip(edades_g,aH3)]

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        ax3.fill_between(spM3.edad_meses, spM3.p25, spM3.p75,
                         alpha=0.12, color=VERDE, label="Rango normal Machos")
        ax3.fill_between(spH3.edad_meses, spH3.p25, spH3.p75,
                         alpha=0.12, color=AZUL, label="Rango normal Hembras")
        ax3.plot(edades_g, pM3, color=VERDE, linewidth=2, label="Predicción Machos")
        ax3.plot(edades_g, pH3, color=AZUL,  linewidth=2, label="Predicción Hembras")
        ax3.axvline(x=edad3, color="#333", linestyle="--", linewidth=1.2, alpha=0.8)
        ax3.scatter([edad3],[peso3], color=AMBER, s=140, zorder=7,
                    edgecolors="#111", linewidths=2, label=f"{peso3:.0f} kg")
        ax3.annotate(f"  {peso3:.0f} kg", xy=(edad3, peso3),
                     xytext=(edad3+0.8, peso3+18),
                     fontsize=9, color=AMBER, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#1a0f00", ec=AMBER, alpha=0.9))
        ax3.set_xlabel("Edad (meses)", fontsize=9)
        ax3.set_ylabel("Peso (kg)", fontsize=9)
        ax3.set_title("Curvas de predicción vs. rangos normales", fontsize=10)
        ax3.legend()
        ax3.set_xlim(0.5, 22.5)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        with st.expander("Ver tabla completa de predicciones"):
            st.dataframe(pd.DataFrame({
                "Mes": edades_g,
                "Machos (kg)":  [round(p) for p in pM3],
                "Hembras (kg)": [round(p) for p in pH3],
                "Alz. M (m)":   [round(a,3) for a in aM3],
                "Alz. H (m)":   [round(a,3) for a in aH3],
            }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIÓN INTERNACIONAL
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-label">Rancho MX vs Literatura internacional</div>',
                unsafe_allow_html=True)

    hintz_m  = {0:55,1:98,2:132,3:170,4:195,5:221,6:245,7:270,
                8:283,9:310,10:318,11:334,12:345,13:359,14:373,15:392,16:415,17:428,18:446}
    hintz_h  = {0:54,1:97,2:131,3:166,4:192,5:212,6:236,7:260,
                8:272,9:296,10:304,11:320,12:329,13:343,14:355,15:375,16:392,17:406,18:424}
    hintz_am = {0:100.6,1:110.8,2:118.5,3:125.2,4:128.9,5:131.6,6:134.6,
                7:137.1,8:139.5,9:141.8,10:142.6,11:144.4,12:145.9,13:147.2,
                14:148.8,15:150.2,16:151.8,17:152.8,18:154.5}
    ker_meses  = [0,1,6,12,18]
    ker_ky     = [67.5,99.3,250.7,353.3,453.9]
    ker_world  = [66.9,98.6,247.1,350.7,444.9]
    ker_aus    = [69.6,102.4,251.4,357.8,460.7]
    ker_alz_ky = [105.7,112.6,135.9,147.8,154.7]
    ker_alz_w  = [106.1,112.0,135.0,147.1,153.8]
    dc_meses = [0,6,12,18]
    dc_m     = [53.8,243.8,337.2,432.7]
    dc_h     = [56.6,248.2,343.6,445.5]
    dc_alz_m = [102.2,135.8,146.5,154.2]
    dc_alz_h = [103.6,136.4,147.6,155.5]
    mx_m_p50  = {0:56,1:99,2:137,3:173,4:207,5:233,6:246,7:273,8:294,9:311,
                 10:322,11:335,12:347,13:360,14:379,15:397,16:412,17:430,18:428}
    mx_h_p50  = {0:56,1:101,2:134,3:169,4:200,5:228,6:245,7:266,8:283,9:298,
                 10:311,11:322,12:332,13:346,14:365,15:384,16:404,17:435,18:444}
    mx_am_p50 = {1:111,2:118,3:123,4:128,5:131,6:134,7:135,8:137,9:139,
                 10:141,11:143,12:144,13:146,14:148,15:149,16:150,17:152,18:153}
    mx_ah_p50 = {1:109,2:117,3:122,4:127,5:130,6:133,7:134,8:136,9:138,
                 10:140,11:142,12:144,13:145,14:147,15:148,16:150,17:151,18:152}

    c1, c2, _ = st.columns([1,1,2])
    with c1:
        var_comp = st.radio("Variable", ["Peso (kg)","Alzada (cm)"], key="var_comp")
    with c2:
        sexo_comp = st.radio("Sexo", ["Machos","Hembras"], key="sexo_comp")

    fig_c, ax_c = plt.subplots(figsize=(12, 5))

    COLS_INT = {
        "MX":    VERDE,
        "Hintz": "#f472b6",
        "KY":    AZUL,
        "Aus":   "#a78bfa",
        "World": "#94a3b8",
        "BR":    AMBER,
    }

    if "Peso" in var_comp:
        if sexo_comp=="Machos":
            ax_c.plot(list(mx_m_p50.keys()), list(mx_m_p50.values()),
                      color=COLS_INT["MX"], lw=3, marker="o", ms=5,
                      label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_m.keys()), list(hintz_m.values()),
                      color=COLS_INT["Hintz"], lw=1.8, linestyle="--", alpha=0.85,
                      label="Hintz 1979 (Canadá)")
            ax_c.plot(ker_meses, ker_ky, color=COLS_INT["KY"], lw=1.8,
                      linestyle="-.", marker="s", ms=6, label="KER — Kentucky")
            ax_c.plot(ker_meses, ker_aus, color=COLS_INT["Aus"], lw=1.5,
                      linestyle=":", marker="^", ms=5, label="KER — Australia")
            ax_c.plot(ker_meses, ker_world, color=COLS_INT["World"], lw=1.8,
                      linestyle="-.", alpha=0.7, label="KER — Mundial")
            ax_c.plot(dc_meses, dc_m, color=COLS_INT["BR"], lw=1.8,
                      marker="D", ms=7, label="De Castro 2021 (Brasil)")
        else:
            ax_c.plot(list(mx_h_p50.keys()), list(mx_h_p50.values()),
                      color=COLS_INT["MX"], lw=3, marker="o", ms=5,
                      label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_h.keys()), list(hintz_h.values()),
                      color=COLS_INT["Hintz"], lw=1.8, linestyle="--", alpha=0.85,
                      label="Hintz 1979 (Canadá)")
            ax_c.plot(ker_meses, ker_ky, color=COLS_INT["KY"], lw=1.8,
                      linestyle="-.", marker="s", ms=6, label="KER — Kentucky")
            ax_c.plot(ker_meses, ker_world, color=COLS_INT["World"], lw=1.8,
                      linestyle="-.", alpha=0.7, label="KER — Mundial")
            ax_c.plot(dc_meses, dc_h, color=COLS_INT["BR"], lw=1.8,
                      marker="D", ms=7, label="De Castro 2021 (Brasil)")
        ax_c.set_ylabel("Peso (kg)", fontsize=9)
        titulo_var = "Peso corporal"
    else:
        if sexo_comp=="Machos":
            ax_c.plot(list(mx_am_p50.keys()), list(mx_am_p50.values()),
                      color=COLS_INT["MX"], lw=3, marker="o", ms=5,
                      label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_am.keys()), list(hintz_am.values()),
                      color=COLS_INT["Hintz"], lw=1.8, linestyle="--", alpha=0.85,
                      label="Hintz 1979 (Canadá)")
            ax_c.plot(ker_meses, ker_alz_ky, color=COLS_INT["KY"], lw=1.8,
                      linestyle="-.", marker="s", ms=6, label="KER — Kentucky")
            ax_c.plot(ker_meses, ker_alz_w, color=COLS_INT["World"], lw=1.8,
                      linestyle="-.", alpha=0.7, label="KER — Mundial")
            ax_c.plot(dc_meses, dc_alz_m, color=COLS_INT["BR"], lw=1.8,
                      marker="D", ms=7, label="De Castro 2021 (Brasil)")
        else:
            ax_c.plot(list(mx_ah_p50.keys()), list(mx_ah_p50.values()),
                      color=COLS_INT["MX"], lw=3, marker="o", ms=5,
                      label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_am.keys()), list(hintz_am.values()),
                      color=COLS_INT["Hintz"], lw=1.8, linestyle="--", alpha=0.85,
                      label="Hintz 1979 (Canadá)")
            ax_c.plot(ker_meses, ker_alz_ky, color=COLS_INT["KY"], lw=1.8,
                      linestyle="-.", marker="s", ms=6, label="KER — Kentucky")
            ax_c.plot(ker_meses, ker_alz_w, color=COLS_INT["World"], lw=1.8,
                      linestyle="-.", alpha=0.7, label="KER — Mundial")
            ax_c.plot(dc_meses, dc_alz_h, color=COLS_INT["BR"], lw=1.8,
                      marker="D", ms=7, label="De Castro 2021 (Brasil)")
        ax_c.set_ylabel("Alzada (cm)", fontsize=9)
        titulo_var = "Alzada a la cruz"

    ax_c.set_xlabel("Edad (meses)", fontsize=9)
    ax_c.set_title(f"{titulo_var} — {sexo_comp} | Rancho MX vs literatura internacional",
                   fontsize=10)
    ax_c.legend(ncol=3)
    ax_c.set_xlim(-0.5, 22)
    plt.tight_layout()
    st.pyplot(fig_c, use_container_width=True)
    plt.close(fig_c)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Tabla comparativa — puntos clave</div>',
                unsafe_allow_html=True)
    meses_k = [0,6,12,18]
    def kv(lst, ml, mt):
        d = dict(zip(ml, lst))
        return [round(d[m],1) if m in d else "—" for m in mt]

    if "Peso" in var_comp and sexo_comp=="Machos":
        dt = {"Mes":meses_k,"Rancho MX":[mx_m_p50.get(m,"—") for m in meses_k],
              "Hintz 1979":[hintz_m.get(m,"—") for m in meses_k],
              "KER Kentucky":kv(ker_ky,ker_meses,meses_k),
              "KER Mundial":kv(ker_world,ker_meses,meses_k),
              "Brasil 2021":kv(dc_m,dc_meses,meses_k)}
    elif "Peso" in var_comp and sexo_comp=="Hembras":
        dt = {"Mes":meses_k,"Rancho MX":[mx_h_p50.get(m,"—") for m in meses_k],
              "Hintz 1979":[hintz_h.get(m,"—") for m in meses_k],
              "KER Kentucky":kv(ker_ky,ker_meses,meses_k),
              "KER Mundial":kv(ker_world,ker_meses,meses_k),
              "Brasil 2021":kv(dc_h,dc_meses,meses_k)}
    elif "Alzada" in var_comp and sexo_comp=="Machos":
        dt = {"Mes":meses_k,"Rancho MX":[mx_am_p50.get(m,"—") for m in meses_k],
              "Hintz 1979":[hintz_am.get(m,"—") for m in meses_k],
              "KER Kentucky":kv(ker_alz_ky,ker_meses,meses_k),
              "KER Mundial":kv(ker_alz_w,ker_meses,meses_k),
              "Brasil 2021":kv(dc_alz_m,dc_meses,meses_k)}
    else:
        dt = {"Mes":meses_k,"Rancho MX":[mx_ah_p50.get(m,"—") for m in meses_k],
              "Hintz 1979":[hintz_am.get(m,"—") for m in meses_k],
              "KER Kentucky":kv(ker_alz_ky,ker_meses,meses_k),
              "KER Mundial":kv(ker_alz_w,ker_meses,meses_k),
              "Brasil 2021":kv(dc_alz_h,dc_meses,meses_k)}

    st.dataframe(pd.DataFrame(dt), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="ind-box" style="margin-top:12px">
        <div class="ind-lbl">Interpretación</div>
        <div class="ind-item ind-green"><div class="ind-dot"></div>
            <span>Al nacimiento el rancho MX es comparable a Canadá (−1 kg) y Brasil (−2.2 kg), pero inferior a Kentucky (−11.5 kg) por selección genética acumulada.</span></div>
        <div class="ind-item ind-green"><div class="ind-dot"></div>
            <span>A los 6 meses las diferencias son menores a 5 kg en todas las poblaciones — el manejo postnatal del rancho es comparable al estándar internacional.</span></div>
        <div class="ind-item ind-amber"><div class="ind-dot"></div>
            <span>La brecha de 17–26 kg a los 18 meses se atribuye a la venta anticipada al hipódromo, no a déficit nutricional.</span></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — REPORTE IA
# ══════════════════════════════════════════════════════════════

with tab5:
    form_ia, result_ia = st.columns([1, 1.4], gap="large")

    with form_ia:
        st.markdown('<div class="section-label">Datos para el reporte</div>',
                    unsafe_allow_html=True)
        nombre_ia = st.text_input("Identificador del potro",
                                  placeholder="Ej. Hijo de Mila Race",
                                  key="nombre_ia")
        c1ia, c2ia = st.columns(2)
        with c1ia:
            sexo_ia = st.radio("Sexo", ["Macho","Hembra"], key="sexo_ia")
        with c2ia:
            rancho_ia = st.text_input("Rancho",
                                      value="Rancho PSI México",
                                      key="rancho_ia")

        st.markdown('<div class="section-label" style="margin-top:12px">Mediciones</div>',
                    unsafe_allow_html=True)
        MESES_IA = [0,1,3,6,9,12,18]
        pesos_ia   = {}
        alzadas_ia = {}
        cols_ia = st.columns(2)
        for i, mes in enumerate(MESES_IA):
            with cols_ia[i % 2]:
                lbl = "Nacimiento" if mes==0 else f"Mes {mes}"
                st.markdown(f"<div style='font-size:10px;color:#4ade80;font-weight:600;"
                            f"margin:8px 0 3px;text-transform:uppercase;letter-spacing:0.4px'>"
                            f"{lbl}</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    pv = st.number_input("kg", min_value=0.0, max_value=700.0,
                                         value=0.0, step=1.0, key=f"pia_{mes}")
                with c2:
                    av = st.number_input("m", min_value=0.0, max_value=2.0,
                                         value=0.0, step=0.01, key=f"aia_{mes}")
                if pv>0: pesos_ia[mes]=pv
                if av>0: alzadas_ia[mes]=av

        st.markdown('<div class="section-label" style="margin-top:12px">Contexto clínico</div>',
                    unsafe_allow_html=True)
        antecedentes = st.text_area("Antecedentes",
                                    placeholder="Desparasitaciones, enfermedades...",
                                    height=70, key="antecedentes_ia")
        manejo = st.text_area("Manejo y alimentación",
                              placeholder="Pastoreo, concentrado...",
                              height=70, key="manejo_ia")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        generar_ia = st.button("Generar reporte clínico →",
                               type="primary", key="btn_ia")

    with result_ia:
        if not generar_ia:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        text-align:center;padding:40px 20px">
                <div style="width:48px;height:48px;border:1px solid #1e1e1e;
                            border-radius:12px;display:flex;align-items:center;
                            justify-content:center;margin:0 auto 16px">
                    <span style="font-size:22px">🤖</span>
                </div>
                <div style="font-size:14px;font-weight:500;color:#555;margin-bottom:6px">
                    Asistente de interpretación clínica
                </div>
                <div style="font-size:12px;color:#333;line-height:1.6">
                    Ingresa los datos del potro y presiona generar.<br>
                    El modelo LLaMA 3.3 70B (Meta AI) via Groq
                    creará un reporte clínico narrativo estructurado.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            meses_con_peso = [m for m in pesos_ia if pesos_ia[m]>0]
            if len(meses_con_peso)<2:
                st.warning("Ingresa al menos 2 mediciones de peso.")
            else:
                sk_ia = "M" if sexo_ia=="Macho" else "H"
                sp_ia = stats_ref[f"stats_{sk_ia}"]
                sa_ia = stats_alz[f"stats_{sk_ia}"]

                datos_eval = []
                for mes in sorted(meses_con_peso):
                    if mes==0:
                        datos_eval.append({"mes":0,"etiqueta":"Nacimiento",
                                           "peso":pesos_ia[0],"diff_pct":0,
                                           "zona_peso":"Peso al nacer","alzada_info":""})
                        continue
                    ref = sp_ia[sp_ia.edad_meses==mes]
                    if ref.empty: continue
                    peso=pesos_ia[mes]
                    p10=ref["p10"].values[0]; p25=ref["p25"].values[0]
                    p50=ref["p50"].values[0]; p75=ref["p75"].values[0]; p90=ref["p90"].values[0]
                    diff=((peso-p50)/p50)*100
                    if peso<p10:    zona="MUY BAJO"
                    elif peso<p25:  zona="BAJO"
                    elif peso<=p75: zona="NORMAL"
                    elif peso<=p90: zona="ALTO"
                    else:           zona="MUY ALTO"
                    alz_info=""
                    if mes in alzadas_ia and alzadas_ia[mes]>0:
                        rfa=sa_ia[sa_ia.edad_meses==mes]
                        if not rfa.empty:
                            a50=rfa["p50"].values[0]
                            da=((alzadas_ia[mes]-a50)/a50)*100
                            alz_info=f"{alzadas_ia[mes]:.2f} m ({da:+.1f}% vs mediana)"
                    datos_eval.append({"mes":mes,"etiqueta":f"Mes {mes}","peso":peso,
                                       "p50":round(p50,1),"diff_pct":round(diff,1),
                                       "zona_peso":zona,"alzada_info":alz_info})

                vals_p=[d["peso"] for d in datos_eval if d["mes"]>0]
                n_bajo=sum(1 for d in datos_eval if "BAJO" in d.get("zona_peso",""))
                n_alto=sum(1 for d in datos_eval if "ALTO" in d.get("zona_peso",""))
                n_eval=len([d for d in datos_eval if d["mes"]>0])
                perdidas=sum(1 for i in range(1,len(vals_p)) if vals_p[i]<vals_p[i-1])
                caida=any((vals_p[i]-vals_p[i-1])/vals_p[i-1]*100<-8
                          for i in range(1,len(vals_p)))

                if (perdidas>=4) or caida: patron_ia="Irregular"
                elif n_eval>0 and n_alto/n_eval>=0.6: patron_ia="Superior"
                elif n_eval>0 and n_bajo/n_eval>=0.6: patron_ia="Inferior"
                else: patron_ia="Normal"

                tabla_mediciones="\n".join([
                    f"  - {d['etiqueta']}: {d['peso']} kg | {d['zona_peso']} "
                    f"| {d['diff_pct']:+.1f}% vs P50 "
                    f"{'| Alzada: '+d['alzada_info'] if d.get('alzada_info') else ''}"
                    if d["mes"]>0 else f"  - Nacimiento: {d['peso']} kg"
                    for d in datos_eval
                ])

                prompt=f"""Eres un medico veterinario especialista en equinos con experiencia en cria de Pura Sangre Ingles (PSI).

Genera un REPORTE CLINICO PROFESIONAL sobre el crecimiento del siguiente potro PSI, con base en mediciones mensuales comparadas contra curvas percentiladas de 217 potros PSI de un rancho mexicano (2015-2025). El rango normal es P25-P75.

DATOS:
- Nombre: {nombre_ia or 'Sin nombre'}
- Sexo: {sexo_ia}
- Rancho: {rancho_ia}
- Patron de crecimiento clasificado: {patron_ia}

MEDICIONES Y PERCENTILES:
{tabla_mediciones}

ANTECEDENTES: {antecedentes if antecedentes else 'No especificados'}
MANEJO: {manejo if manejo else 'No especificado'}

Estructura el reporte con estas secciones claramente marcadas:
1. RESUMEN EJECUTIVO (2-3 oraciones con el hallazgo principal)
2. EVALUACION DEL CRECIMIENTO (analisis del patron, tendencia, comparacion con referencia)
3. HALLAZGOS RELEVANTES (meses con alertas, correlacion peso-alzada si aplica)
4. RECOMENDACIONES CLINICAS (minimo 3, ordenadas de mayor a menor urgencia)
5. PLAN DE SEGUIMIENTO (frecuencia, indicadores de alarma, proxima evaluacion)

Escribe en espanol con terminologia veterinaria apropiada y comprensible para el personal del rancho."""

                with st.spinner("Generando reporte con LLaMA 3.3 70B..."):
                    try:
                        api_key = os.environ.get("GROQ_API_KEY","")
                        if not api_key:
                            st.error("Configura GROQ_API_KEY en los secretos de Streamlit.")
                            st.stop()

                        resp = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Content-Type":"application/json",
                                     "Authorization":f"Bearer {api_key}"},
                            json={"model":"llama-3.3-70b-versatile",
                                  "messages":[
                                      {"role":"system","content":
                                       "Eres un medico veterinario especialista en equinos. "
                                       "Generas reportes clinicos profesionales en espanol."},
                                      {"role":"user","content":prompt}
                                  ],
                                  "temperature":0.4,"max_tokens":1500}
                        )
                        data = resp.json()

                        if "choices" in data and len(data["choices"])>0:
                            reporte = data["choices"][0]["message"]["content"]

                            cls_rep = {"Normal":"patron-normal","Superior":"patron-superior",
                                       "Inferior":"patron-inferior","Irregular":"patron-irregular"}
                            st.markdown(
                                f'<div class="patron-chip {cls_rep.get(patron_ia,"patron-normal")}"'
                                f' style="margin-bottom:12px">Patrón {patron_ia}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div style="background:#111;border:1px solid #1e1e1e;'
                                f'border-radius:10px;padding:16px 18px;font-size:13px;'
                                f'color:#ccc;line-height:1.7;white-space:pre-wrap">'
                                f'{reporte}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                            st.download_button(
                                label="Descargar reporte .txt",
                                data=(f"REPORTE CLINICO CRECIPSI\n{'='*50}\n"
                                      f"Potro: {nombre_ia or 'Sin nombre'}\n"
                                      f"Sexo: {sexo_ia}\nRancho: {rancho_ia}\n"
                                      f"Patron: {patron_ia}\n"
                                      f"Modelo: LLaMA 3.3 70B via Groq\n{'='*50}\n\n"
                                      f"{reporte}"),
                                file_name=f"reporte_{(nombre_ia or 'potro').replace(' ','_')}.txt",
                                mime="text/plain"
                            )
                        elif "error" in data:
                            st.error(f"Error Groq: {data['error'].get('message','')}")
                        else:
                            st.error("Sin respuesta del modelo.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════
# TAB 6 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════

with tab6:
    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown('<div class="section-label">Base de datos</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Registros zootécnicos reales de un rancho PSI mexicano (2015–2025).
        **217 animales** · **4,175 mediciones de peso** · **3,981 de alzada**.
        Completitud del 100% en alzada para meses 1–22.
        El 89.4% de los animales tienen peso al nacer registrado.
        """)

        st.markdown('<div class="section-label" style="margin-top:16px">Estadística aplicada</div>',
                    unsafe_allow_html=True)
        for item in [
            "Percentiles P10–P90 por edad (0–22 meses) y sexo",
            "Regresión polinomial grado 3 — variables: sexo, edad, alzada",
            "Validación train/test 80%/20% (random_state=42)",
            "Clasificador basado en criterios clínicos equinos",
            "Correlación peso–alzada: r=0.9666 (Pearson, p<0.001)",
        ]:
            st.markdown(f"""
            <div class="ind-item" style="margin-bottom:6px">
                <div class="ind-dot" style="background:#4ade80"></div>
                <span style="font-size:13px;color:#aaa">{item}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:16px">Módulo LLM</div>',
                    unsafe_allow_html=True)
        for item in [
            "Tipo: Open source, gratuito, sin restricciones regionales",
            "Funcion: Interpretacion clinica narrativa automatizada",
            "El LLM asiste al veterinario, no lo sustituye",
        ]:
            st.markdown(f"""
            <div class="ind-item ind-blue" style="margin-bottom:6px">
                <div class="ind-dot"></div>
                <span style="font-size:13px">{item}</span>
            </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="section-label">Metricas de validacion</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Modelo":  ["Peso (con alzada)","Peso (sin alzada)","Alzada"],
            "R2":      ["0.9641","0.9458","0.9552"],
            "MAE":     ["15.1 kg","19.6 kg","2.0 cm"],
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label" style="margin-top:16px">Distribucion de patrones</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Patron":    ["Normal","Superior","Inferior","Irregular"],
            "N":         [143, 37, 34, 3],
            "Porcentaje":["65.9%","17.1%","15.7%","1.4%"],
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label" style="margin-top:16px">Referencias</div>',
                    unsafe_allow_html=True)
        refs = [
            "Hintz HF et al. J Anim Sci. 1979;48(3):480-487.",
            "Brown-Douglas CG, Pagan JD. Adv Eq Nutr IV. 2009:213-220.",
            "De Castro LL et al. Int J Plant Anim Environ Sci. 2021;11(3):352-362.",
            "NRC. Nutrient Requirements of Horses. 6th ed. NAP; 2007.",
            "James G et al. Introduction to Statistical Learning. Springer; 2021.",
            "Dohoo I et al. Veterinary Epidemiologic Research. VER Inc; 2009.",
        ]
        for i, r in enumerate(refs):
            st.markdown(
                f'<div style="display:flex;gap:8px;margin-bottom:5px">' +
                f'<span style="font-size:11px;color:#333;font-family:monospace;flex-shrink:0;margin-top:2px">{i+1}.</span>' +
                f'<span style="font-size:12px;color:#666;line-height:1.5">{r}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="alert-ok" style="margin-top:16px;font-size:12px">' +
            'Salgado Alvarez C. CreciPSI. FMVZ-UNAM. Diplomado IA en Salud Global. 2026.' +
            '</div>',
            unsafe_allow_html=True
        )
