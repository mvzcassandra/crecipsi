# ══════════════════════════════════════════════════════════════
# CreciPSI v7.0 — Diseño Médico Profesional (Opción C)
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

# ── CSS MÉDICO PROFESIONAL ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--color-border-secondary); border-radius: 3px; }

/* ── Header ── */
.ehr-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: var(--color-background-primary);
    border-bottom: 1px solid var(--color-border-tertiary);
}
.ehr-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.ehr-logo-icon {
    width: 32px; height: 32px;
    background: var(--color-background-secondary);
    border: 1px solid var(--color-border-secondary);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.ehr-logo-text { font-size: 15px; font-weight: 600; color: var(--color-text-primary); }
.ehr-logo-sub  { font-size: 11px; color: var(--color-text-secondary); margin-top: 1px; }
.ehr-badges { display: flex; gap: 6px; }
.ehr-badge {
    background: var(--color-background-secondary);
    border: 1px solid var(--color-border-tertiary);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 11px;
    color: var(--color-text-secondary);
}

/* ── Metrics strip ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    border-bottom: 1px solid var(--color-border-tertiary);
    background: var(--color-background-primary);
}
.metric-cell {
    padding: 10px 16px;
    border-right: 1px solid var(--color-border-tertiary);
}
.metric-cell:last-child { border-right: none; }
.metric-val { font-size: 18px; font-weight: 600; color: var(--color-text-primary); line-height: 1; }
.metric-lbl { font-size: 10px; color: var(--color-text-secondary); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.4px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--color-background-primary) !important;
    border-bottom: 1px solid var(--color-border-tertiary) !important;
    padding: 0 20px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--color-text-secondary) !important;
    padding: 9px 14px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--color-text-primary) !important;
    border-bottom-color: var(--color-text-primary) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important;
    background: var(--color-background-tertiary) !important;
}

/* ── Layout EHR: sidebar + content ── */
.ehr-layout {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 0;
    min-height: calc(100vh - 120px);
}
.ehr-sidebar {
    background: var(--color-background-primary);
    border-right: 1px solid var(--color-border-tertiary);
    padding: 16px;
    overflow-y: auto;
}
.ehr-content {
    background: var(--color-background-tertiary);
    padding: 20px;
    overflow-y: auto;
}

/* ── Section labels ── */
.section-lbl {
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 14px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--color-border-tertiary);
}
.section-lbl:first-child { margin-top: 0; }

/* ── Field rows ── */
.field-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--color-border-tertiary);
}
.field-lbl { font-size: 11px; color: var(--color-text-secondary); }
.field-val { font-size: 12px; font-weight: 500; color: var(--color-text-primary); }

/* ── Cards ── */
.card {
    background: var(--color-background-primary);
    border: 1px solid var(--color-border-tertiary);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

/* ── Patron chips ── */
.chip-normal    { background: var(--color-background-success); color: var(--color-text-success); border: 1px solid var(--color-border-success); }
.chip-superior  { background: var(--color-background-info);    color: var(--color-text-info);    border: 1px solid var(--color-border-info); }
.chip-inferior  { background: var(--color-background-warning); color: var(--color-text-warning); border: 1px solid var(--color-border-warning); }
.chip-irregular { background: var(--color-background-danger);  color: var(--color-text-danger);  border: 1px solid var(--color-border-danger); }
.patron-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 10px 0;
}
.stat-cell {
    background: var(--color-background-secondary);
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
    border: 1px solid var(--color-border-tertiary);
}
.stat-val { font-size: 17px; font-weight: 600; color: var(--color-text-primary); line-height: 1; }
.stat-lbl { font-size: 10px; color: var(--color-text-secondary); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.3px; }
.stat-cell.ok  .stat-val { color: var(--color-text-success); }
.stat-cell.warn .stat-val { color: var(--color-text-warning); }
.stat-cell.bad  .stat-val { color: var(--color-text-danger); }
.stat-cell.info .stat-val { color: var(--color-text-info); }

/* ── Indicaciones ── */
.ind-box {
    background: var(--color-background-secondary);
    border: 1px solid var(--color-border-tertiary);
    border-radius: 7px;
    padding: 10px 12px;
    margin-top: 10px;
}
.ind-title { font-size: 10px; font-weight: 600; color: var(--color-text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 7px; }
.ind-item  { display: flex; gap: 7px; font-size: 12px; color: var(--color-text-secondary); margin-bottom: 5px; align-items: flex-start; line-height: 1.5; }
.ind-item:last-child { margin-bottom: 0; }
.ind-dot   { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; margin-top: 6px; background: var(--color-border-secondary); }
.ind-success .ind-dot { background: var(--color-text-success); }
.ind-success { color: var(--color-text-success); }
.ind-warning .ind-dot { background: var(--color-text-warning); }
.ind-warning { color: var(--color-text-warning); }
.ind-danger  .ind-dot { background: var(--color-text-danger); }
.ind-danger  { color: var(--color-text-danger); }
.ind-info    .ind-dot { background: var(--color-text-info); }
.ind-info    { color: var(--color-text-info); }

/* ── Alert strips ── */
.strip-ok   { background: var(--color-background-success); border: 1px solid var(--color-border-success); border-radius: 6px; padding: 7px 11px; font-size: 12px; color: var(--color-text-success); margin: 6px 0; }
.strip-warn { background: var(--color-background-warning); border: 1px solid var(--color-border-warning); border-radius: 6px; padding: 7px 11px; font-size: 12px; color: var(--color-text-warning); margin: 6px 0; }
.strip-info { background: var(--color-background-info);    border: 1px solid var(--color-border-info);    border-radius: 6px; padding: 7px 11px; font-size: 12px; color: var(--color-text-info);    margin: 6px 0; }

/* ── Month cells for sidebar ── */
.month-cell {
    background: var(--color-background-secondary);
    border: 1px solid var(--color-border-tertiary);
    border-radius: 6px;
    padding: 7px 9px;
    margin-bottom: 5px;
}
.month-cell.has-data { border-color: var(--color-border-success); background: var(--color-background-success); }
.month-cell-lbl { font-size: 10px; color: var(--color-text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 4px; }
.month-cell.has-data .month-cell-lbl { color: var(--color-text-success); }

/* ── Inputs ── */
.stNumberInput > div > div > input,
.stTextInput  > div > div > input {
    font-size: 13px !important;
    background: var(--color-background-primary) !important;
    border: 1px solid var(--color-border-secondary) !important;
    border-radius: 6px !important;
    color: var(--color-text-primary) !important;
}
.stNumberInput > div > div > input { text-align: center !important; }
.stTextInput  > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--color-text-primary) !important;
    box-shadow: 0 0 0 2px var(--color-border-secondary) !important;
}
.stTextArea > div > div > textarea {
    font-size: 13px !important;
    background: var(--color-background-primary) !important;
    border: 1px solid var(--color-border-secondary) !important;
    border-radius: 6px !important;
    color: var(--color-text-primary) !important;
}
.stRadio > div { flex-direction: row !important; gap: 6px !important; flex-wrap: wrap !important; }
.stRadio > div > label {
    background: var(--color-background-secondary) !important;
    border: 1px solid var(--color-border-secondary) !important;
    border-radius: 5px !important;
    padding: 4px 11px !important;
    font-size: 12px !important;
    color: var(--color-text-secondary) !important;
    cursor: pointer !important;
}
.stRadio > div > label:has(input:checked) {
    border-color: var(--color-text-primary) !important;
    color: var(--color-text-primary) !important;
    font-weight: 500 !important;
    background: var(--color-background-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--color-text-primary) !important;
    color: var(--color-background-primary) !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    width: 100% !important;
    padding: 9px 16px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Slider ── */
.stSlider > div > div > div { background: var(--color-border-secondary) !important; }

/* ── Labels ── */
.stTextInput label, .stNumberInput label,
.stTextArea label, .stRadio > label, .stSlider > label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--color-text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    font-size: 12px !important;
    color: var(--color-text-secondary) !important;
    background: var(--color-background-secondary) !important;
    border: 1px solid var(--color-border-tertiary) !important;
    border-radius: 6px !important;
}

/* ── Divider ── */
.div { height: 1px; background: var(--color-border-tertiary); margin: 12px 0; }

/* ── Result header ── */
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}
.result-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
}
.result-meta { font-size: 11px; color: var(--color-text-secondary); }

/* ── Pred box ── */
.pred-box {
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    border: 1px solid var(--color-border-tertiary);
    margin-bottom: 12px;
}
.pred-val { font-size: 32px; font-weight: 700; line-height: 1; }
.pred-lbl { font-size: 11px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB: estilo limpio claro/oscuro ────────────────────
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#666666",
    "axes.titlecolor":   "#333333",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "xtick.color":       "#999999",
    "ytick.color":       "#999999",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#dddddd",
    "legend.fontsize":   9,
    "text.color":        "#333333",
    "font.family":       "sans-serif",
})
C = {"M": "#0f3460", "H": "#831843"}
AMBER = "#d97706"


# ── CARGAR MODELOS ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar():
    import joblib

    # ── Percentiles desde CSV (master limpio 4,184 registros) ──
    pp = pd.read_csv("percentiles_peso.csv")
    pt = pd.read_csv("percentiles_talla.csv")

    # Convertir a dict {stats_M: DataFrame, stats_H: DataFrame}
    # compatible con el resto del código
    sr = {
        "stats_M": pp[pp["sexo"] == "M"].reset_index(drop=True),
        "stats_H": pp[pp["sexo"] == "H"].reset_index(drop=True),
    }
    sa = {
        "stats_M": pt[pt["sexo"] == "M"].reset_index(drop=True),
        "stats_H": pt[pt["sexo"] == "H"].reset_index(drop=True),
    }

    # ── Modelos nuevos (pipeline con scaler integrado) ──────────
    # M2: Peso ~ edad_meses + talla_cm + sexo_num  [PRINCIPAL]
    # Features en orden: [edad_meses, talla_cm, sexo_num]
    mp = joblib.load("modelo_peso_alzada.pkl")

    # M3: Talla ~ edad_meses + sexo_num
    # Features en orden: [edad_meses, sexo_num]
    ma = joblib.load("modelo_talla.pkl")

    # M1: Peso ~ edad_meses + sexo_num (sin talla, fallback)
    mp_simple = joblib.load("modelo_peso_simple.pkl")

    return sr, sa, mp, ma, mp_simple

try:
    stats_ref, stats_alz, mod_peso, mod_alz, mod_peso_simple = cargar()
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")
    st.stop()


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="ehr-header">
  <div class="ehr-logo">
    <div class="ehr-logo-icon">🐴</div>
    <div>
      <div class="ehr-logo-text">CreciPSI</div>
      <div class="ehr-logo-sub">Monitor inteligente de crecimiento equino</div>
    </div>
  </div>
  <div class="ehr-badges">
    <span class="ehr-badge">FMVZ · UNAM</span>
    <span class="ehr-badge">DIASG 2025–2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MÉTRICAS ─────────────────────────────────────────────────
st.markdown("""
<div class="metrics-row">
  <div class="metric-cell"><div class="metric-val">217</div><div class="metric-lbl">Potros PSI</div></div>
  <div class="metric-cell"><div class="metric-val">4,184</div><div class="metric-lbl">Mediciones</div></div>
  <div class="metric-cell"><div class="metric-val">11 años</div><div class="metric-lbl">2015 – 2025</div></div>
  <div class="metric-cell"><div class="metric-val">0.966</div><div class="metric-lbl">R² modelo</div></div>
  <div class="metric-cell"><div class="metric-val">±14.8 kg</div><div class="metric-lbl">Error medio</div></div>
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
# HELPER: fig curvas
# ══════════════════════════════════════════════════════════════
def fig_curvas(stats, color, titulo, ylabel, ylim,
               meses_anot, fmt_anot, offset_anot,
               datos_potro=None, nombre_potro=None,
               punto_pred=None, edad_pred=None,
               w=11, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    edades = stats["edad_meses"]

    ax.fill_between(edades, stats.p10, stats.p90, alpha=0.06, color=color)
    ax.fill_between(edades, stats.p25, stats.p75, alpha=0.18, color=color,
                    label="Rango normal P25–P75")
    ax.plot(edades, stats.p50, color=color, linewidth=2,
            label="Mediana P50", zorder=3)
    ax.plot(edades, stats.p10, color=color, linewidth=0.8,
            linestyle=":", alpha=0.35)
    ax.plot(edades, stats.p90, color=color, linewidth=0.8,
            linestyle=":", alpha=0.35, label="P10 / P90")

    for mes in meses_anot:
        f = stats[stats.edad_meses == mes]
        if len(f) == 0: continue
        v = f["p50"].values[0]
        ax.annotate(fmt_anot.format(v),
                    xy=(mes, v), xytext=(mes + 0.6, v + offset_anot),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.9))

    if datos_potro and len(datos_potro) >= 2:
        eds = sorted(datos_potro.keys())
        vls = [datos_potro[e] for e in eds]
        ax.plot(eds, vls, color=AMBER, linewidth=2.2,
                marker="o", markersize=6,
                label=nombre_potro or "Potro evaluado", zorder=5)
        for e, v in zip(eds, vls):
            ax.annotate(f"{v:.0f}" if offset_anot > 5 else f"{v:.2f}",
                        xy=(e, v),
                        xytext=(e + 0.3, v + offset_anot * 0.5),
                        fontsize=7.5, color=AMBER,
                        arrowprops=dict(arrowstyle="-", color=AMBER,
                                        lw=0.7, alpha=0.5))

    if punto_pred is not None and edad_pred is not None:
        ax.axvline(x=edad_pred, color="#aaaaaa", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.scatter([edad_pred], [punto_pred], color=AMBER, s=160,
                   zorder=7, edgecolors="white", linewidths=2,
                   label=f"Pred: {punto_pred:.0f}" if offset_anot > 5
                         else f"Pred: {punto_pred:.3f}")
        ax.annotate(f"  {punto_pred:.0f} kg" if offset_anot > 5
                    else f"  {punto_pred:.3f} m",
                    xy=(edad_pred, punto_pred),
                    xytext=(edad_pred + 0.8, punto_pred + offset_anot * 1.1),
                    fontsize=9, color=AMBER, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#fffbeb", ec=AMBER, alpha=0.95))

    ax.set_xlabel("Edad (meses)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=10, pad=8)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    xlim_min = -0.3 if 0 in (edades.values
                              if hasattr(edades, "values") else edades) else 0.5
    ax.set_xlim(xlim_min, 22.5)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 1 — CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════
with tab1:
    sb1, ct1 = st.columns([1, 3], gap="small")

    with sb1:
        st.markdown('<div class="section-lbl">Filtros</div>', unsafe_allow_html=True)
        sx1 = st.radio("Sexo", ["Machos", "Hembras"], key="sx1")
        vr1 = st.radio("Variable", ["Peso (kg)", "Alzada (cm)"], key="vr1")
        sk1 = "M" if sx1 == "Machos" else "H"
        n1  = 113 if sk1 == "M" else 104
        st.markdown(
            f'<div class="strip-info" style="margin-top:12px">' +
            f'<strong>n = {n1}</strong> animales · ' +
            f'{"Machos" if sk1=="M" else "Hembras"} PSI</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="section-lbl">Referencia rápida</div>', unsafe_allow_html=True)
        st_d1 = (stats_ref[f"stats_{sk1}"] if "Peso" in vr1
                 else stats_alz[f"stats_{sk1}"])
        t1 = st_d1[["edad_meses","p25","p50","p75"]].copy()
        t1.columns = ["Mes","P25","P50","P75"]
        for _, row in t1[t1["Mes"].isin([0,6,12,18])].iterrows():
            dec = 0 if "Peso" in vr1 else 0
            st.markdown(
                f'<div class="field-row">' +
                f'<span class="field-lbl">Mes {int(row["Mes"])}</span>' +
                f'<span class="field-val">{row["P50"]:.{dec}f}</span>' +
                f'</div>',
                unsafe_allow_html=True
            )

    with ct1:
        st.markdown(
            f'<div style="padding:16px 20px 0">' +
            f'<span style="font-size:14px;font-weight:600;color:var(--color-text-primary)">' +
            f'{"Peso corporal" if "Peso" in vr1 else "Alzada a la cruz"} — ' +
            f'{"Machos" if sk1=="M" else "Hembras"} PSI (n={n1})' +
            f'</span></div>',
            unsafe_allow_html=True
        )
        if "Peso" in vr1:
            fig1 = fig_curvas(
                stats_ref[f"stats_{sk1}"], C[sk1],
                "", "Peso (kg)", (20, 570),
                [0, 6, 12, 18], "{:.0f} kg", 20, w=9, h=4.2
            )
        else:
            fig1 = fig_curvas(
                stats_alz[f"stats_{sk1}"], C[sk1],
                "", "Alzada (cm)", (85, 175),
                [6, 12, 18], "{:.0f} cm", 2, w=9, h=4.2
            )
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown('<div style="padding:0 20px">', unsafe_allow_html=True)
        with st.expander("Ver tabla completa de valores"):
            t_full = st_d1[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
            t_full.columns = ["Edad","P10","P25","P50","P75","P90","N"]
            st.dataframe(t_full.round(1),
                         use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="strip-info" style="margin:10px 0">P25–P75 es el rango normal — ' +
            'contiene el 50% central de la población. ' +
            'Valores fuera de P10–P90 justifican evaluación clínica.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUAR POTRO
# ══════════════════════════════════════════════════════════════
with tab2:
    MESES_CLAVE = [1, 3, 6, 9, 12, 18]
    MESES_EXTRA = [2,4,5,7,8,10,11,13,14,15,16,17,19,20,21,22]

    sb2, ct2 = st.columns([1, 2.2], gap="small")

    # ── SIDEBAR FORMULARIO ──────────────────────────────────
    with sb2:
        st.markdown('<div style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Identificación</div>', unsafe_allow_html=True)
        nombre2 = st.text_input("Nombre del potro",
                                placeholder="Ej. Hijo de Mila Race",
                                key="nombre2")
        sx2 = st.radio("Sexo", ["Macho", "Hembra"], key="sx2")
        pnac2 = st.number_input("Peso al nacer (kg)",
                                min_value=0.0, max_value=80.0,
                                value=0.0, step=0.5, key="pnac2")
        st.markdown('<div class="section-lbl">Mediciones mensuales</div>',
                    unsafe_allow_html=True)
        st.caption("Deja en 0 los meses sin dato. Mínimo 2 meses con peso.")

        pesos2   = {}
        alzadas2 = {}
        if pnac2 > 0: pesos2[0] = pnac2

        for mes in MESES_CLAVE:
            has = pesos2.get(mes, 0) > 0 or alzadas2.get(mes, 0) > 0
            st.markdown(
                f'<div style="font-size:10px;font-weight:600;color:var(--color-text-secondary);' +
                f'text-transform:uppercase;letter-spacing:0.4px;margin:8px 0 3px">Mes {mes}</div>',
                unsafe_allow_html=True
            )
            ca, cb = st.columns(2)
            with ca:
                pv = st.number_input("kg", min_value=0.0, max_value=700.0,
                                     value=0.0, step=1.0, key=f"p2_{mes}")
            with cb:
                av = st.number_input("cm", min_value=0.0, max_value=200.0,
                                     value=0.0, step=0.5, key=f"a2_{mes}")
            if pv > 0: pesos2[mes] = pv
            if av > 0: alzadas2[mes] = av

        with st.expander("+ Meses adicionales"):
            for mes in MESES_EXTRA:
                st.markdown(
                    f'<div style="font-size:10px;color:var(--color-text-secondary);' +
                    f'margin:6px 0 2px">Mes {mes}</div>',
                    unsafe_allow_html=True
                )
                ca2, cb2 = st.columns(2)
                with ca2:
                    pv2 = st.number_input("kg", min_value=0.0, max_value=700.0,
                                          value=0.0, step=1.0, key=f"p2_{mes}")
                with cb2:
                    av2 = st.number_input("cm", min_value=0.0, max_value=200.0,
                                          value=0.0, step=0.5, key=f"a2_{mes}")
                if pv2 > 0: pesos2[mes] = pv2
                if av2 > 0: alzadas2[mes] = av2

        st.markdown('<div style="margin-top:12px">', unsafe_allow_html=True)
        analizar2 = st.button("Analizar crecimiento", type="primary", key="btn2")
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── CONTENIDO RESULTADO ─────────────────────────────────
    with ct2:
        st.markdown('<div style="padding:16px 20px">', unsafe_allow_html=True)

        if not analizar2:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:60px 20px;text-align:center">
              <div style="width:52px;height:52px;background:var(--color-background-secondary);
                          border:1px solid var(--color-border-tertiary);border-radius:12px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:24px;margin-bottom:16px">🐴</div>
              <div style="font-size:14px;font-weight:500;color:var(--color-text-primary);
                          margin-bottom:6px">Sin datos aún</div>
              <div style="font-size:12px;color:var(--color-text-secondary);
                          max-width:320px;line-height:1.6">
                Completa los datos del potro en el panel izquierdo
                y presiona <strong>Analizar crecimiento</strong>.
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
                    p10 = ref["p10"].values[0]; p25 = ref["p25"].values[0]
                    p50 = ref["p50"].values[0]; p75 = ref["p75"].values[0]
                    p90 = ref["p90"].values[0]
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
                            elif alzadas2[mes] <= rfa["p75"].values[0]: alz_zona="Normal"
                            else:                                        alz_zona="Alta"
                    filas2.append({"mes":mes,"peso":peso,"p10":p10,"p25":p25,
                                   "p50":p50,"p75":p75,"p90":p90,"diff":diff,
                                   "zona":zona,"alerta":alerta,
                                   "alzada":alzadas2.get(mes),"alz_zona":alz_zona})

                n_f = len(filas2)
                pct_alto = alto_cnt/n_f if n_f>0 else 0
                pct_bajo = bajo_cnt/n_f if n_f>0 else 0
                vals_p   = [f["peso"] for f in filas2]
                perdidas = sum(1 for i in range(1,len(vals_p)) if vals_p[i]<vals_p[i-1])
                caida    = any((vals_p[i]-vals_p[i-1])/vals_p[i-1]*100<-8
                               for i in range(1,len(vals_p)))

                if (perdidas >= 4) or caida:
                    patron2="Patrón Irregular"; cls_p="chip-irregular"
                    inds=[("danger","Evaluación clínica urgente — pérdida de peso detectada."),
                          ("danger","Descartar parasitosis, enfermedad GI o estrés."),
                          ("warning","Revisar calidad y cantidad del alimento."),
                          ("warning","Verificar acceso a agua limpia y comedero.")]
                elif pct_alto >= 0.60:
                    patron2="Patrón Superior"; cls_p="chip-superior"
                    inds=[("info","Crecimiento excelente — por encima del P75 en la mayoría de meses."),
                          ("","Mantener el plan nutricional y de manejo actual."),
                          ("warning","Vigilar condición corporal para evitar sobrepeso tardío.")]
                elif pct_bajo >= 0.60:
                    patron2="Patrón Inferior"; cls_p="chip-inferior"
                    inds=[("warning","Revisar aporte energético — incrementar concentrado."),
                          ("warning","Evaluar desparasitación — alta carga reduce absorción."),
                          ("warning","Verificar forraje disponible y acceso a agua limpia."),
                          ("","Repetir evaluación en 4 semanas tras ajuste nutricional.")]
                else:
                    patron2="Patrón Normal"; cls_p="chip-normal"
                    inds=[("success","Mantener el programa de manejo y alimentación actual."),
                          ("success","Continuar con pesajes mensuales para seguimiento.")]
                    if alertas2:
                        inds.append(("warning", f"Vigilar meses con alerta: {alertas2}."))

                # Header resultado
                st.markdown(
                    f'<div class="result-header">' +
                    f'<div><div class="result-name">{nombre_d}</div>' +
                    f'<div class="result-meta">{sx2} · {len(meses_p)} mediciones evaluadas</div></div>' +
                    f'<span class="patron-chip {cls_p}">{patron2}</span>' +
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Stats
                ganancia = round(vals_p[-1]-vals_p[0]) if len(vals_p)>=2 else 0
                norm_pct = round(sum(1 for f in filas2 if f["zona"]=="NORMAL")/n_f*100)
                cls_n = "ok" if norm_pct>=60 else ("warn" if norm_pct>=40 else "bad")
                cls_a = "ok" if len(alertas2)==0 else ("warn" if len(alertas2)<=2 else "bad")
                st.markdown(f"""
                <div class="stat-grid">
                  <div class="stat-cell {cls_n}">
                    <div class="stat-val">{norm_pct}%</div>
                    <div class="stat-lbl">En rango</div>
                  </div>
                  <div class="stat-cell">
                    <div class="stat-val">+{ganancia} kg</div>
                    <div class="stat-lbl">Ganancia total</div>
                  </div>
                  <div class="stat-cell {cls_a}">
                    <div class="stat-val">{len(alertas2)}</div>
                    <div class="stat-lbl">Alertas</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Alerta strip
                if len(alertas2)==0:
                    st.markdown(
                        '<div class="strip-ok">Sin alertas — crecimiento dentro del rango en todos los meses</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="strip-warn">Alertas en meses: {alertas2}</div>',
                        unsafe_allow_html=True)

                # Indicaciones
                ind_html = "".join([
                    f'<div class="ind-item ind-{c if c else ""}">' +
                    f'<div class="ind-dot"></div><span>{t}</span></div>'
                    for c,t in inds
                ])
                st.markdown(
                    f'<div class="ind-box"><div class="ind-title">Indicaciones clínicas</div>' +
                    f'{ind_html}</div>',
                    unsafe_allow_html=True
                )

                # Gráficas
                st.markdown('<div class="div"></div>', unsafe_allow_html=True)
                pesos_dict = {f["mes"]:f["peso"] for f in filas2}
                ylim_p = (50, max(max(pesos_dict.values())+60, 520))
                fig_p = fig_curvas(
                    sp2, C[sk2],
                    f"Peso — {nombre_d} vs. curvas de referencia",
                    "Peso (kg)", ylim_p,
                    [6,12,18], "{:.0f} kg", 20,
                    datos_potro=pesos_dict, nombre_potro=nombre_d,
                    w=9, h=3.8
                )
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)

                alz_dict = {m:alzadas2[m] for m in alzadas2 if alzadas2[m]>0 and m>0}
                if len(alz_dict)>=2:
                    fig_a = fig_curvas(
                        sa2, C[sk2],
                        f"Alzada — {nombre_d} vs. curvas de referencia",
                        "Alzada (cm)", (85, 180),
                        [6,12,18], "{:.0f} cm", 2,
                        datos_potro=alz_dict, nombre_potro=nombre_d,
                        w=9, h=3.2
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
                        "Alzada(cm)":f["alzada"] if f["alzada"] else "—",
                        "Est. alzada":f["alz_zona"] if f["alz_zona"] else "—",
                    } for f in filas2])
                    def ce(v):
                        if "BAJO" in str(v): return "color:#92400e"
                        elif "ALTO" in str(v): return "color:#1e40af"
                        elif "NORMAL" in str(v): return "color:#166534"
                        return ""
                    try: styled=df2.style.map(ce, subset=["Estado"])
                    except AttributeError: styled=df2.style.applymap(ce, subset=["Estado"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab3:
    sb3, ct3 = st.columns([1, 2.2], gap="small")

    with sb3:
        st.markdown('<div style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Parámetros</div>', unsafe_allow_html=True)
        sx3   = st.radio("Sexo", ["Macho","Hembra"], key="sx3")
        edad3 = st.slider("Edad (meses)", 1, 22, 6, key="edad3")
        alz3  = st.number_input("Alzada actual (cm) — opcional",
                                min_value=0.0, max_value=200.0,
                                value=0.0, step=0.5, key="alz3")
        if alz3>0:
            st.markdown('<div class="strip-ok">Modelo principal · R²=0.9664</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="strip-info">Sin alzada · R²=0.9540</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ct3:
        st.markdown('<div style="padding:16px 20px">', unsafe_allow_html=True)
        sk3   = "M" if sx3=="Macho" else "H"
        sbin3 = 1 if sk3=="M" else 0
        sp3   = stats_ref[f"stats_{sk3}"]
        sa3   = stats_alz[f"stats_{sk3}"]
        ref_a3= sa3[sa3.edad_meses==edad3]
        # Alzada de referencia en cm (P50 del mes correspondiente)
        alz_m3= ref_a3["p50"].values[0] if len(ref_a3)>0 else 135.0
        alz_u3= alz3 if alz3>0 else alz_m3
        # Modelo principal M2: features en orden [edad_meses, talla_cm, sexo_num]
        peso3 = mod_peso.predict([[edad3, alz_u3, sbin3]])[0]
        # Modelo M3: features en orden [edad_meses, sexo_num] → predice cm
        alzp3 = mod_alz.predict([[edad3, sbin3]])[0]
        ref_p3= sp3[sp3.edad_meses==edad3]

        if len(ref_p3)>0:
            p25r=ref_p3["p25"].values[0]; p50r=ref_p3["p50"].values[0]
            p75r=ref_p3["p75"].values[0]
            if peso3<p25r:    pos="Inferior al rango"; cls_pred="chip-inferior"
            elif peso3<=p75r: pos="Dentro del rango";  cls_pred="chip-normal"
            else:             pos="Superior al rango"; cls_pred="chip-superior"

            c_res1, c_res2 = st.columns([1,1])
            with c_res1:
                st.markdown(f"""
                <div class="pred-box" style="background:var(--color-background-secondary)">
                    <div style="font-size:11px;color:var(--color-text-secondary);
                                text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">
                        Peso predicho · mes {edad3}
                    </div>
                    <div class="pred-val" style="color:var(--color-text-primary)">{peso3:.0f} kg</div>
                    <span class="patron-chip {cls_pred}" style="margin-top:8px;display:inline-flex">{pos}</span>
                </div>
                """, unsafe_allow_html=True)
            with c_res2:
                if len(ref_a3)>0:
                    a50=ref_a3["p50"].values[0]
                    st.markdown(f"""
                    <div class="pred-box" style="background:var(--color-background-secondary)">
                        <div style="font-size:11px;color:var(--color-text-secondary);
                                    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">
                            Alzada predicha · mes {edad3}
                        </div>
                        <div class="pred-val" style="color:var(--color-text-primary)">{alzp3:.1f} cm</div>
                        <div style="font-size:11px;color:var(--color-text-secondary);margin-top:6px">
                            Mediana rancho: {a50:.1f} cm
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-grid" style="grid-template-columns:repeat(3,1fr)">
              <div class="stat-cell"><div class="stat-val">{p25r:.0f}</div><div class="stat-lbl">P25 rancho</div></div>
              <div class="stat-cell info"><div class="stat-val">{p50r:.0f}</div><div class="stat-lbl">P50 rancho</div></div>
              <div class="stat-cell"><div class="stat-val">{p75r:.0f}</div><div class="stat-lbl">P75 rancho</div></div>
            </div>
            """, unsafe_allow_html=True)

        # Gráfica predictor — usar P50 de talla como covariable
        edades_g=list(range(0,23))
        saM3=stats_alz["stats_M"]; saH3=stats_alz["stats_H"]
        spM3=stats_ref["stats_M"]; spH3=stats_ref["stats_H"]
        # Alzada P50 en cm por mes (fallback si el mes no tiene dato)
        aM3=[saM3[saM3.edad_meses==e]["p50"].values[0] if len(saM3[saM3.edad_meses==e])>0 else 135.0 for e in edades_g]
        aH3=[saH3[saH3.edad_meses==e]["p50"].values[0] if len(saH3[saH3.edad_meses==e])>0 else 132.0 for e in edades_g]
        # Modelo M2: [edad, talla_cm, sexo_num]
        pM3=[mod_peso.predict([[e,a,1]])[0] for e,a in zip(edades_g,aM3)]
        pH3=[mod_peso.predict([[e,a,0]])[0] for e,a in zip(edades_g,aH3)]

        fig3, ax3 = plt.subplots(figsize=(9,4))
        fig3.patch.set_alpha(0); ax3.patch.set_alpha(0)
        ax3.fill_between(spM3.edad_meses, spM3.p25, spM3.p75, alpha=0.1, color=C["M"], label="Rango M")
        ax3.fill_between(spH3.edad_meses, spH3.p25, spH3.p75, alpha=0.1, color=C["H"], label="Rango H")
        ax3.plot(edades_g,pM3,color=C["M"],lw=2,label="Machos")
        ax3.plot(edades_g,pH3,color=C["H"],lw=2,label="Hembras")
        ax3.axvline(x=edad3,color="#aaa",lw=1.2,linestyle="--",alpha=0.7)
        ax3.scatter([edad3],[peso3],color=AMBER,s=140,zorder=7,edgecolors="white",lw=2,
                    label=f"{peso3:.0f} kg")
        ax3.annotate(f"  {peso3:.0f} kg",xy=(edad3,peso3),xytext=(edad3+0.8,peso3+18),
                     fontsize=9,color=AMBER,fontweight="bold",
                     arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.3",fc="#fffbeb",ec=AMBER,alpha=0.95))
        ax3.set_xlabel("Edad (meses)",fontsize=9)
        ax3.set_ylabel("Peso (kg)",fontsize=9)
        ax3.set_title("Predicciones vs. rangos normales",fontsize=10)
        ax3.legend(); ax3.set_xlim(-0.3,22.5)
        plt.tight_layout()
        st.pyplot(fig3,use_container_width=True); plt.close(fig3)

        with st.expander("Ver tabla completa de predicciones"):
            st.dataframe(pd.DataFrame({
                "Mes":edades_g,
                "Machos(kg)":[round(p) for p in pM3],
                "Hembras(kg)":[round(p) for p in pH3],
                "Alz.M(cm)":[round(a,1) for a in aM3],
                "Alz.H(cm)":[round(a,1) for a in aH3],
            }), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIÓN INTERNACIONAL
# ══════════════════════════════════════════════════════════════
with tab4:
    sb4, ct4 = st.columns([1,3], gap="small")

    hintz_m  = {0:55,1:98,2:132,3:170,4:195,5:221,6:245,7:270,8:283,9:310,10:318,11:334,12:345,13:359,14:373,15:392,16:415,17:428,18:446}
    hintz_h  = {0:54,1:97,2:131,3:166,4:192,5:212,6:236,7:260,8:272,9:296,10:304,11:320,12:329,13:343,14:355,15:375,16:392,17:406,18:424}
    hintz_am = {0:100.6,1:110.8,2:118.5,3:125.2,4:128.9,5:131.6,6:134.6,7:137.1,8:139.5,9:141.8,10:142.6,11:144.4,12:145.9,13:147.2,14:148.8,15:150.2,16:151.8,17:152.8,18:154.5}
    ker_meses=[0,1,6,12,18]; ker_ky=[67.5,99.3,250.7,353.3,453.9]; ker_world=[66.9,98.6,247.1,350.7,444.9]
    ker_aus=[69.6,102.4,251.4,357.8,460.7]; ker_alz_ky=[105.7,112.6,135.9,147.8,154.7]; ker_alz_w=[106.1,112.0,135.0,147.1,153.8]
    dc_meses=[0,6,12,18]; dc_m=[53.8,243.8,337.2,432.7]; dc_h=[56.6,248.2,343.6,445.5]
    dc_alz_m=[102.2,135.8,146.5,154.2]; dc_alz_h=[103.6,136.4,147.6,155.5]
    mx_m_p50={0:56,1:99,2:137,3:173,4:207,5:233,6:246,7:273,8:294,9:311,10:322,11:335,12:347,13:360,14:379,15:397,16:412,17:430,18:428}
    mx_h_p50={0:56,1:101,2:134,3:169,4:200,5:228,6:245,7:266,8:283,9:298,10:311,11:322,12:332,13:346,14:365,15:384,16:404,17:435,18:444}
    mx_am_p50={1:111,2:118,3:123,4:128,5:131,6:134,7:135,8:137,9:139,10:141,11:143,12:144,13:146,14:148,15:149,16:150,17:152,18:153}
    mx_ah_p50={1:109,2:117,3:122,4:127,5:130,6:133,7:134,8:136,9:138,10:140,11:142,12:144,13:145,14:147,15:148,16:150,17:151,18:152}

    with sb4:
        st.markdown('<div style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Filtros</div>', unsafe_allow_html=True)
        var_comp  = st.radio("Variable",["Peso (kg)","Alzada (cm)"], key="var_comp")
        sexo_comp = st.radio("Sexo",["Machos","Hembras"], key="sexo_comp")
        st.markdown('<div class="section-lbl">Diferencias (vs MX)</div>', unsafe_allow_html=True)
        meses_ref=[0,6,12,18]
        fuentes={"Canada":{0:55,6:245,12:345,18:446},"Kentucky":{0:67.5,6:250.7,12:353.3,18:453.9},"Brasil":{0:53.8,6:243.8,12:337.2,18:432.7}}
        mx_ref={0:56,6:246,12:347,18:430}
        for mes in meses_ref:
            st.markdown(f'<div style="font-size:10px;font-weight:600;color:var(--color-text-secondary);margin:8px 0 3px;text-transform:uppercase">Mes {mes}</div>', unsafe_allow_html=True)
            for fuente, vals in fuentes.items():
                diff = vals.get(mes,0) - mx_ref.get(mes,0)
                col = "var(--color-text-success)" if diff>=0 else "var(--color-text-danger)"
                st.markdown(f'<div class="field-row"><span class="field-lbl">{fuente}</span><span style="font-size:12px;font-weight:500;color:{col}">{diff:+.0f} kg</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ct4:
        st.markdown('<div style="padding:16px 20px">', unsafe_allow_html=True)
        COLS_INT={"MX":"#0f3460","Hintz":"#db2777","KY":"#2563eb","Aus":"#7c3aed","World":"#64748b","BR":"#d97706"}
        fig_c, ax_c = plt.subplots(figsize=(9,4.5))
        fig_c.patch.set_alpha(0); ax_c.patch.set_alpha(0)

        def kv(lst,ml,mt):
            d=dict(zip(ml,lst)); return [round(d[m],1) if m in d else None for m in mt]

        if "Peso" in var_comp:
            if sexo_comp=="Machos":
                ax_c.plot(list(mx_m_p50.keys()),list(mx_m_p50.values()),color=COLS_INT["MX"],lw=3,marker="o",ms=5,label="Rancho MX",zorder=6)
                ax_c.plot(list(hintz_m.keys()),list(hintz_m.values()),color=COLS_INT["Hintz"],lw=1.8,linestyle="--",label="Hintz 1979 (Canada)")
                ax_c.plot(ker_meses,ker_ky,color=COLS_INT["KY"],lw=1.8,linestyle="-.",marker="s",ms=6,label="KER Kentucky")
                ax_c.plot(ker_meses,ker_aus,color=COLS_INT["Aus"],lw=1.5,linestyle=":",marker="^",ms=5,label="KER Australia")
                ax_c.plot(ker_meses,ker_world,color=COLS_INT["World"],lw=1.8,linestyle="-.",alpha=0.8,label="KER Mundial")
                ax_c.plot(dc_meses,dc_m,color=COLS_INT["BR"],lw=1.8,marker="D",ms=7,label="De Castro 2021 (Brasil)")
            else:
                ax_c.plot(list(mx_h_p50.keys()),list(mx_h_p50.values()),color=COLS_INT["MX"],lw=3,marker="o",ms=5,label="Rancho MX",zorder=6)
                ax_c.plot(list(hintz_h.keys()),list(hintz_h.values()),color=COLS_INT["Hintz"],lw=1.8,linestyle="--",label="Hintz 1979 (Canada)")
                ax_c.plot(ker_meses,ker_ky,color=COLS_INT["KY"],lw=1.8,linestyle="-.",marker="s",ms=6,label="KER Kentucky")
                ax_c.plot(ker_meses,ker_world,color=COLS_INT["World"],lw=1.8,linestyle="-.",alpha=0.8,label="KER Mundial")
                ax_c.plot(dc_meses,dc_h,color=COLS_INT["BR"],lw=1.8,marker="D",ms=7,label="De Castro 2021 (Brasil)")
            ax_c.set_ylabel("Peso (kg)",fontsize=9)
        else:
            if sexo_comp=="Machos":
                ax_c.plot(list(mx_am_p50.keys()),list(mx_am_p50.values()),color=COLS_INT["MX"],lw=3,marker="o",ms=5,label="Rancho MX",zorder=6)
                ax_c.plot(list(hintz_am.keys()),list(hintz_am.values()),color=COLS_INT["Hintz"],lw=1.8,linestyle="--",label="Hintz 1979")
                ax_c.plot(ker_meses,ker_alz_ky,color=COLS_INT["KY"],lw=1.8,linestyle="-.",marker="s",ms=6,label="KER Kentucky")
                ax_c.plot(ker_meses,ker_alz_w,color=COLS_INT["World"],lw=1.8,linestyle="-.",alpha=0.8,label="KER Mundial")
                ax_c.plot(dc_meses,dc_alz_m,color=COLS_INT["BR"],lw=1.8,marker="D",ms=7,label="De Castro 2021")
            else:
                ax_c.plot(list(mx_ah_p50.keys()),list(mx_ah_p50.values()),color=COLS_INT["MX"],lw=3,marker="o",ms=5,label="Rancho MX",zorder=6)
                ax_c.plot(list(hintz_am.keys()),list(hintz_am.values()),color=COLS_INT["Hintz"],lw=1.8,linestyle="--",label="Hintz 1979")
                ax_c.plot(ker_meses,ker_alz_ky,color=COLS_INT["KY"],lw=1.8,linestyle="-.",marker="s",ms=6,label="KER Kentucky")
                ax_c.plot(ker_meses,ker_alz_w,color=COLS_INT["World"],lw=1.8,linestyle="-.",alpha=0.8,label="KER Mundial")
                ax_c.plot(dc_meses,dc_alz_h,color=COLS_INT["BR"],lw=1.8,marker="D",ms=7,label="De Castro 2021")
            ax_c.set_ylabel("Alzada (cm)",fontsize=9)

        ax_c.set_xlabel("Edad (meses)",fontsize=9)
        ax_c.set_title(f"{'Peso' if 'Peso' in var_comp else 'Alzada'} — {sexo_comp} | Rancho MX vs literatura internacional",fontsize=10)
        ax_c.legend(ncol=3,fontsize=9); ax_c.set_xlim(-0.5,22)
        plt.tight_layout()
        st.pyplot(fig_c,use_container_width=True); plt.close(fig_c)

        meses_k=[0,6,12,18]
        def kv2(lst,ml,mt):
            d=dict(zip(ml,lst)); return [round(d[m],1) if m in d else "—" for m in mt]

        if "Peso" in var_comp and sexo_comp=="Machos":
            dt={"Mes":meses_k,"Rancho MX":[mx_m_p50.get(m,"—") for m in meses_k],"Hintz 1979":[hintz_m.get(m,"—") for m in meses_k],"KER Kentucky":kv2(ker_ky,ker_meses,meses_k),"KER Mundial":kv2(ker_world,ker_meses,meses_k),"Brasil 2021":kv2(dc_m,dc_meses,meses_k)}
        elif "Peso" in var_comp and sexo_comp=="Hembras":
            dt={"Mes":meses_k,"Rancho MX":[mx_h_p50.get(m,"—") for m in meses_k],"Hintz 1979":[hintz_h.get(m,"—") for m in meses_k],"KER Kentucky":kv2(ker_ky,ker_meses,meses_k),"KER Mundial":kv2(ker_world,ker_meses,meses_k),"Brasil 2021":kv2(dc_h,dc_meses,meses_k)}
        elif "Alzada" in var_comp and sexo_comp=="Machos":
            dt={"Mes":meses_k,"Rancho MX":[mx_am_p50.get(m,"—") for m in meses_k],"Hintz 1979":[hintz_am.get(m,"—") for m in meses_k],"KER Kentucky":kv2(ker_alz_ky,ker_meses,meses_k),"KER Mundial":kv2(ker_alz_w,ker_meses,meses_k),"Brasil 2021":kv2(dc_alz_m,dc_meses,meses_k)}
        else:
            dt={"Mes":meses_k,"Rancho MX":[mx_ah_p50.get(m,"—") for m in meses_k],"Hintz 1979":[hintz_am.get(m,"—") for m in meses_k],"KER Kentucky":kv2(ker_alz_ky,ker_meses,meses_k),"KER Mundial":kv2(ker_alz_w,ker_meses,meses_k),"Brasil 2021":kv2(dc_alz_h,dc_meses,meses_k)}
        st.dataframe(pd.DataFrame(dt),use_container_width=True,hide_index=True)

        st.markdown("""
        <div class="ind-box" style="margin-top:10px">
          <div class="ind-title">Interpretación</div>
          <div class="ind-item ind-success"><div class="ind-dot"></div><span>Al nacimiento el rancho MX es comparable a Canadá (−1 kg) y Brasil (−2.2 kg). La diferencia con Kentucky (−11.5 kg) se explica por presión genética acumulada.</span></div>
          <div class="ind-item ind-success"><div class="ind-dot"></div><span>A los 6 meses las diferencias son menores a 5 kg en todas las poblaciones — el manejo postnatal del rancho es comparable al estándar internacional.</span></div>
          <div class="ind-item ind-warning"><div class="ind-dot"></div><span>La brecha de 17–26 kg a los 18 meses se atribuye a la venta anticipada al hipódromo, no a déficit nutricional.</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — REPORTE IA
# ══════════════════════════════════════════════════════════════
with tab5:
    sb5, ct5 = st.columns([1,2.2], gap="small")

    with sb5:
        st.markdown('<div style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Identificación</div>', unsafe_allow_html=True)
        nombre_ia = st.text_input("Nombre del potro", placeholder="Ej. Hijo de Mila Race", key="nombre_ia")
        c1ia,c2ia=st.columns(2)
        with c1ia: sexo_ia=st.radio("Sexo",["Macho","Hembra"],key="sexo_ia")
        with c2ia: rancho_ia=st.text_input("Rancho",value="Rancho PSI Mexico",key="rancho_ia")

        st.markdown('<div class="section-lbl">Mediciones</div>', unsafe_allow_html=True)
        MESES_IA=[0,1,3,6,9,12,18]; pesos_ia={}; alzadas_ia={}
        for mes in MESES_IA:
            lbl="Nacimiento" if mes==0 else f"Mes {mes}"
            st.markdown(f'<div style="font-size:10px;font-weight:600;color:var(--color-text-secondary);margin:7px 0 2px;text-transform:uppercase">{lbl}</div>', unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: pv=st.number_input("kg",min_value=0.0,max_value=700.0,value=0.0,step=1.0,key=f"pia_{mes}")
            with c2: av=st.number_input("m",min_value=0.0,max_value=2.0,value=0.0,step=0.01,key=f"aia_{mes}")
            if pv>0: pesos_ia[mes]=pv
            if av>0: alzadas_ia[mes]=av

        st.markdown('<div class="section-lbl">Contexto clínico</div>', unsafe_allow_html=True)
        antecedentes=st.text_area("Antecedentes",placeholder="Desparasitaciones, enfermedades...",height=65,key="antecedentes_ia")
        manejo=st.text_area("Manejo",placeholder="Pastoreo, concentrado...",height=65,key="manejo_ia")
        st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
        generar_ia=st.button("Generar reporte clínico",type="primary",key="btn_ia")
        st.markdown('</div></div>', unsafe_allow_html=True)

    with ct5:
        st.markdown('<div style="padding:16px 20px">', unsafe_allow_html=True)
        if not generar_ia:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:60px 20px;text-align:center">
              <div style="width:52px;height:52px;background:var(--color-background-secondary);
                          border:1px solid var(--color-border-tertiary);border-radius:12px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:24px;margin-bottom:16px">🤖</div>
              <div style="font-size:14px;font-weight:500;color:var(--color-text-primary);margin-bottom:6px">
                Asistente de interpretación clínica
              </div>
              <div style="font-size:12px;color:var(--color-text-secondary);max-width:320px;line-height:1.6">
                Ingresa los datos del potro y presiona <strong>Generar reporte clínico</strong>.
                El modelo LLaMA 3.3 70B (Meta AI) vía Groq generará un reporte narrativo estructurado.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            mcp=[m for m in pesos_ia if pesos_ia[m]>0]
            if len(mcp)<2:
                st.warning("Ingresa al menos 2 mediciones de peso.")
            else:
                sk_ia="M" if sexo_ia=="Macho" else "H"
                sp_ia=stats_ref[f"stats_{sk_ia}"]; sa_ia=stats_alz[f"stats_{sk_ia}"]
                datos_eval=[]
                for mes in sorted(mcp):
                    if mes==0:
                        datos_eval.append({"mes":0,"etiqueta":"Nacimiento","peso":pesos_ia[0],"diff_pct":0,"zona_peso":"Peso al nacer","alzada_info":""})
                        continue
                    ref=sp_ia[sp_ia.edad_meses==mes]
                    if ref.empty: continue
                    peso=pesos_ia[mes]; p50=ref["p50"].values[0]; p25=ref["p25"].values[0]
                    p10=ref["p10"].values[0]; p75=ref["p75"].values[0]; p90=ref["p90"].values[0]
                    diff=((peso-p50)/p50)*100
                    if peso<p10: zona="MUY BAJO"
                    elif peso<p25: zona="BAJO"
                    elif peso<=p75: zona="NORMAL"
                    elif peso<=p90: zona="ALTO"
                    else: zona="MUY ALTO"
                    alz_info=""
                    if mes in alzadas_ia and alzadas_ia[mes]>0:
                        rfa=sa_ia[sa_ia.edad_meses==mes]
                        if not rfa.empty:
                            a50=rfa["p50"].values[0]; da=((alzadas_ia[mes]-a50)/a50)*100
                            alz_info=f"{alzadas_ia[mes]:.1f} cm ({da:+.1f}% vs mediana)"
                    datos_eval.append({"mes":mes,"etiqueta":f"Mes {mes}","peso":peso,"p50":round(p50,1),"diff_pct":round(diff,1),"zona_peso":zona,"alzada_info":alz_info})

                vals_p=[d["peso"] for d in datos_eval if d["mes"]>0]
                n_bajo=sum(1 for d in datos_eval if "BAJO" in d.get("zona_peso",""))
                n_alto=sum(1 for d in datos_eval if "ALTO" in d.get("zona_peso",""))
                n_eval=len([d for d in datos_eval if d["mes"]>0])
                perdidas=sum(1 for i in range(1,len(vals_p)) if vals_p[i]<vals_p[i-1])
                caida=any((vals_p[i]-vals_p[i-1])/vals_p[i-1]*100<-8 for i in range(1,len(vals_p)))
                if (perdidas>=4) or caida: patron_ia="Irregular"
                elif n_eval>0 and n_alto/n_eval>=0.6: patron_ia="Superior"
                elif n_eval>0 and n_bajo/n_eval>=0.6: patron_ia="Inferior"
                else: patron_ia="Normal"

                tabla_med=chr(10).join([
                    f"  - {d['etiqueta']}: {d['peso']} kg | {d['zona_peso']} | {d['diff_pct']:+.1f}% vs P50{'| Alzada: '+d['alzada_info'] if d.get('alzada_info') else ''}"
                    if d["mes"]>0 else f"  - Nacimiento: {d['peso']} kg"
                    for d in datos_eval
                ])

                prompt=f"""Eres un medico veterinario especialista en equinos con experiencia en cria de Pura Sangre Ingles (PSI).

Genera un REPORTE CLINICO PROFESIONAL. El potro fue evaluado contra curvas percentiladas de 217 potros PSI de un rancho mexicano (2015-2025). Rango normal: P25-P75.

DATOS:
- Nombre: {nombre_ia or 'Sin nombre'} | Sexo: {sexo_ia} | Rancho: {rancho_ia}
- Patron: {patron_ia}

MEDICIONES:
{tabla_med}

ANTECEDENTES: {antecedentes if antecedentes else 'No especificados'}
MANEJO: {manejo if manejo else 'No especificado'}

Estructura el reporte con:
1. RESUMEN EJECUTIVO (2-3 oraciones)
2. EVALUACION DEL CRECIMIENTO
3. HALLAZGOS RELEVANTES
4. RECOMENDACIONES CLINICAS (minimo 3, por urgencia)
5. PLAN DE SEGUIMIENTO

Escribe en espanol con terminologia veterinaria apropiada."""

                with st.spinner("Generando reporte con LLaMA 3.3 70B..."):
                    try:
                        api_key=os.environ.get("GROQ_API_KEY","")
                        if not api_key:
                            st.error("Configura GROQ_API_KEY en Streamlit Secrets.")
                            st.stop()
                        resp=requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},
                            json={"model":"llama-3.3-70b-versatile",
                                  "messages":[{"role":"system","content":"Eres un medico veterinario especialista en equinos. Generas reportes clinicos profesionales en espanol."},
                                              {"role":"user","content":prompt}],
                                  "temperature":0.4,"max_tokens":1500}
                        )
                        data=resp.json()
                        if "choices" in data and len(data["choices"])>0:
                            reporte=data["choices"][0]["message"]["content"]
                            cls_r={"Normal":"chip-normal","Superior":"chip-superior","Inferior":"chip-inferior","Irregular":"chip-irregular"}
                            st.markdown(
                                f'<div class="result-header">' +
                                f'<div><div class="result-name">{nombre_ia or "Potro evaluado"}</div>' +
                                f'<div class="result-meta">{sexo_ia} · {rancho_ia}</div></div>' +
                                f'<span class="patron-chip {cls_r.get(patron_ia,"chip-normal")}">Patron {patron_ia}</span>' +
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div style="background:var(--color-background-secondary);border:1px solid var(--color-border-tertiary);border-radius:8px;padding:16px;font-size:13px;color:var(--color-text-primary);line-height:1.8;white-space:pre-wrap">{reporte}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
                            st.download_button(
                                label="Descargar reporte .txt",
                                data=(f"REPORTE CLINICO CRECIPSI\n{'='*50}\nPotro: {nombre_ia or 'Sin nombre'}\nSexo: {sexo_ia}\nRancho: {rancho_ia}\nPatron: {patron_ia}\nModelo: LLaMA 3.3 70B via Groq\n{'='*50}\n\n{reporte}"),
                                file_name=f"reporte_{(nombre_ia or 'potro').replace(' ','_')}.txt",
                                mime="text/plain"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif "error" in data:
                            st.error(f"Error Groq: {data['error'].get('message','')}")
                        else:
                            st.error("Sin respuesta del modelo.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════
with tab6:
    sb6, ct6 = st.columns([1,2.2], gap="small")

    with sb6:
        st.markdown('<div style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Métricas de validación</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Modelo":["M2: Peso+talla+sexo","M1: Peso+sexo","M3: Talla+sexo"],
            "R² CV":["0.9658","0.9540","0.9612"],
            "MAE CV":["14.9 kg","17.8 kg","1.8 cm"],
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-lbl">Dataset</div>', unsafe_allow_html=True)
        for lbl,val in [("Animales","217"),("Mediciones peso","4,184"),("Mediciones talla","3,990"),("Con peso al nacer","100%"),("Periodo","2015–2025"),("Machos / Hembras","113 / 104")]:
            st.markdown(f'<div class="field-row"><span class="field-lbl">{lbl}</span><span class="field-val">{val}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-lbl">Distribución patrones</div>', unsafe_allow_html=True)
        for pat,n,pct in [("Normal",144,"66.4%"),("Superior",37,"17.1%"),("Inferior",33,"15.2%"),("Irregular",3,"1.4%")]:
            st.markdown(f'<div class="field-row"><span class="field-lbl">{pat}</span><span class="field-val">{n} ({pct})</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ct6:
        st.markdown('<div style="padding:16px 20px">', unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Metodología estadística</div>', unsafe_allow_html=True)
        for item in [
            "Percentiles P10–P90 por edad (0–23 meses) y sexo — calculados directamente del master (n=4,184), NumPy con interpolación lineal",
            "Regresión polinomial grado 3 — scikit-learn Pipeline(PolynomialFeatures, StandardScaler, LinearRegression) — scaler integrado en pipeline para evitar pérdida en despliegue",
            "M2 (principal): peso ~ edad + talla(cm) + sexo; M1 (fallback): peso ~ edad + sexo; M3: talla ~ edad + sexo",
            "Validación cruzada 5-fold (KFold, random_state=42) — R² CV=0.9658, MAE CV=14.9 kg para M2",
            "Master construido desde 11 archivos Excel con fechas de nacimiento exactas — edad calculada con relativedelta (python-dateutil)",
            "Clasificador de patrón basado en criterios clínicos percentilados — reglas interpretables y calibradas",
            "Correlación peso–talla: r=0.9664 (Pearson, p<0.001) — justifica inclusión de talla como covariable en M2",
        ]:
            st.markdown(
                f'<div class="ind-item" style="margin-bottom:8px">' +
                f'<div class="ind-dot" style="background:var(--color-text-primary)"></div>' +
                f'<span style="font-size:13px;color:var(--color-text-secondary);line-height:1.5">{item}</span>' +
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-lbl" style="margin-top:16px">Módulo LLM</div>', unsafe_allow_html=True)
        for item in [
            "Modelo: LLaMA 3.3 70B (Meta AI) vía Groq API — open source, gratuito, sin restricciones regionales",
            "Prompt estructurado con datos del paciente, percentiles y contexto clínico",
            "El LLM actúa como asistente de interpretación — la responsabilidad clínica recae en el veterinario",
        ]:
            st.markdown(
                f'<div class="ind-item ind-info" style="margin-bottom:8px">' +
                f'<div class="ind-dot"></div>' +
                f'<span style="font-size:13px;line-height:1.5">{item}</span>' +
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-lbl" style="margin-top:16px">Referencias (Vancouver)</div>', unsafe_allow_html=True)
        refs=[
            "1. Hintz HF et al. J Anim Sci. 1979;48(3):480-487.",
            "2. Brown-Douglas CG, Pagan JD. Adv Eq Nutr IV. 2009:213-220.",
            "3. De Castro LL et al. Int J Plant Anim Environ Sci. 2021;11(3):352-362.",
            "4. NRC. Nutrient Requirements of Horses. 6th ed. NAP; 2007.",
            "5. James G et al. Introduction to Statistical Learning. Springer; 2021.",
            "6. Dohoo I et al. Veterinary Epidemiologic Research. VER Inc; 2009.",
        ]
        for r in refs:
            st.markdown(f'<div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:4px;line-height:1.5">{r}</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="strip-info" style="margin-top:16px;font-size:12px">' +
            'Salgado Alvarez C. CreciPSI. FMVZ-UNAM. Diplomado IA en Salud Global. 2026.' +
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
