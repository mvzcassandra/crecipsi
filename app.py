# ══════════════════════════════════════════════════════════════
# CreciPSI v5.2 — Versión completa corregida
# FMVZ-UNAM | Diplomado IA en Salud Global 2025-2026
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CreciPSI",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
#MainMenu,footer,.stDeployButton{display:none}
.header{
    background:linear-gradient(135deg,#0d3320 0%,#1a4731 60%,#2d6a4a 100%);
    padding:1.5rem 2rem;border-radius:14px;
    color:white;margin-bottom:1.5rem;
    display:flex;align-items:center;justify-content:space-between;
}
.header h1{font-size:1.8rem;font-weight:700;margin:0;letter-spacing:-0.5px}
.header p{font-size:0.85rem;opacity:0.75;margin:0.2rem 0 0}
.badge{
    display:inline-block;background:rgba(255,255,255,0.15);
    padding:0.25rem 0.85rem;border-radius:20px;font-size:0.72rem;
    border:0.5px solid rgba(255,255,255,0.25);
}
.metric-strip{
    display:grid;grid-template-columns:repeat(5,1fr);
    gap:0.75rem;margin-bottom:1.5rem;
}
.metric-box{
    background:#f8faf8;border:0.5px solid #d4e8d8;
    border-radius:10px;padding:0.85rem 1rem;text-align:center;
}
.metric-box .mv{font-size:1.5rem;font-weight:700;color:#1a4731;line-height:1}
.metric-box .ml{font-size:0.7rem;color:#6b8f72;text-transform:uppercase;
                letter-spacing:0.4px;margin-top:0.25rem}
.section-pill{
    display:inline-flex;align-items:center;gap:0.4rem;
    background:#e8f4ee;border:0.5px solid #1a4731;
    border-radius:20px;padding:0.25rem 0.85rem;
    font-size:0.75rem;font-weight:600;color:#1a4731;margin-bottom:0.75rem;
}
.patron-box{
    border-radius:10px;padding:0.85rem 1.2rem;
    font-size:1rem;font-weight:600;text-align:center;margin:0.6rem 0;
}
.p-normal{background:#e8f4ee;color:#1a4731;border:1.5px solid #52b788}
.p-superior{background:#dbeafe;color:#1e40af;border:1.5px solid #60a5fa}
.p-inferior{background:#fff3e0;color:#92400e;border:1.5px solid #fbbf24}
.p-irregular{background:#fee2e2;color:#991b1b;border:1.5px solid #f87171}
.ind-box{
    background:#f0faf4;border:0.5px solid #52b788;
    border-radius:10px;padding:1rem 1.2rem;margin:0.75rem 0;
}
.ind-titulo{font-size:0.72rem;font-weight:700;color:#1a4731;
            text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.6rem}
.ind-item{display:flex;gap:0.6rem;font-size:0.83rem;
          color:#2d5a3d;margin-bottom:0.4rem;align-items:flex-start}
.ind-item:last-child{margin-bottom:0}
.ind-dot{width:6px;height:6px;border-radius:50%;
         background:#1a4731;flex-shrink:0;margin-top:5px}
.ind-warn .ind-dot{background:#d97706}.ind-warn{color:#7c4a00}
.ind-alert .ind-dot{background:#dc2626}.ind-alert{color:#7f1d1d}
.ind-info .ind-dot{background:#2563eb}.ind-info{color:#1e3a8a}
.ok-strip{background:#f0fdf4;border:0.5px solid #86efac;border-radius:8px;
          padding:0.6rem 1rem;font-size:0.82rem;color:#14532d;margin:0.5rem 0}
.warn-strip{background:#fffbeb;border:0.5px solid #fde68a;border-radius:8px;
            padding:0.6rem 1rem;font-size:0.82rem;color:#78350f;margin:0.5rem 0}
.stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin:0.75rem 0}
.stat-card{background:#f8faf8;border-radius:8px;padding:0.75rem;text-align:center;
           border:0.5px solid #d4e8d8}
.stat-card .sv{font-size:1.3rem;font-weight:700;color:#1a4731}
.stat-card .sl{font-size:0.7rem;color:#6b8f72;text-transform:uppercase;
               letter-spacing:0.3px;margin-top:2px}
.stat-card.warn .sv{color:#d97706}
.stat-card.bad .sv{color:#dc2626}
.stTabs [data-baseweb="tab"]{font-size:0.9rem;font-weight:600;padding:0.5rem 1.2rem}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:#1a4731}
.stTabs [data-baseweb="tab-highlight"]{background-color:#1a4731}
hr{border:none;border-top:0.5px solid #d4e8d8;margin:1.2rem 0}
</style>
""", unsafe_allow_html=True)


# ── CARGAR MODELOS ───────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
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

C = {"M": "#1a4731", "H": "#831843"}


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div>
    <h1>🐴 CreciPSI</h1>
    <p>Monitor Inteligente de Crecimiento · Pura Sangre Inglés</p>
  </div>
  <div style="text-align:right">
    <span class="badge">FMVZ-UNAM</span><br>
    <span style="font-size:0.72rem;opacity:0.65;margin-top:4px;display:block">
      Diplomado IA en Salud Global · 2025-2026
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-strip">
  <div class="metric-box"><div class="mv">217</div><div class="ml">Potros PSI</div></div>
  <div class="metric-box"><div class="mv">4,175</div><div class="ml">Mediciones</div></div>
  <div class="metric-box"><div class="mv">10 años</div><div class="ml">2015-2025</div></div>
  <div class="metric-box"><div class="mv">0.964</div><div class="ml">R² Modelo</div></div>
  <div class="metric-box"><div class="mv">15 kg</div><div class="ml">Error medio</div></div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Curvas de Referencia",
    "🔍 Evaluar un Potro",
    "🎯 Predictor",
    "🌍 Comparación Internacional",
    "🤖 Reporte IA",
    "ℹ️ Metodología"
])


# ══════════════════════════════════════════════════════════════
# HELPER: graficar curvas
# ══════════════════════════════════════════════════════════════

def fig_curvas(stats, color, titulo, ylabel, ylim,
               meses_anotados, fmt_anot, offset_anot,
               datos_potro=None, nombre_potro=None,
               punto_pred=None, edad_pred=None):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor("#f8faf8")
    ax.set_facecolor("#f8faf8")
    edades = stats["edad_meses"]

    ax.fill_between(edades, stats.p10, stats.p90, alpha=0.08, color=color)
    ax.fill_between(edades, stats.p25, stats.p75, alpha=0.22, color=color,
                    label="Rango normal (P25-P75)")
    ax.plot(edades, stats.p50, color=color, linewidth=2.2,
            label="Mediana (P50)", zorder=3)
    ax.plot(edades, stats.p10, color=color, linewidth=0.8,
            linestyle=":", alpha=0.45, label="P10 / P90")
    ax.plot(edades, stats.p90, color=color, linewidth=0.8,
            linestyle=":", alpha=0.45)

    for mes in meses_anotados:
        f = stats[stats.edad_meses == mes]
        if len(f) == 0:
            continue
        v = f["p50"].values[0]
        ax.annotate(
            fmt_anot.format(v),
            xy=(mes, v), xytext=(mes + 0.5, v + offset_anot),
            fontsize=8, color=color, fontweight="600",
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85)
        )

    if datos_potro and len(datos_potro) >= 2:
        eds = sorted(datos_potro.keys())
        vls = [datos_potro[e] for e in eds]
        ax.plot(eds, vls, color="#e67e22", linewidth=2.5,
                marker="o", markersize=7,
                label=nombre_potro or "Potro evaluado", zorder=5)
        for e, v in zip(eds, vls):
            ax.annotate(
                f"{v:.0f}" if offset_anot > 5 else f"{v:.2f}",
                xy=(e, v), xytext=(e + 0.3, v + offset_anot * 0.6),
                fontsize=7.5, color="#c0392b", fontweight="600",
                arrowprops=dict(arrowstyle="-", color="#e67e22", lw=0.8, alpha=0.6)
            )

    if punto_pred is not None and edad_pred is not None:
        ax.axvline(x=edad_pred, color="#94a3b8", linestyle="--",
                   linewidth=1.2, alpha=0.8)
        ax.scatter([edad_pred], [punto_pred], color="#e67e22", s=180,
                   zorder=6, edgecolors="white", linewidths=2,
                   label=f"Predicción: {punto_pred:.0f}" if offset_anot > 5
                         else f"Predicción: {punto_pred:.3f}")
        ax.annotate(
            f"  {punto_pred:.0f} kg" if offset_anot > 5 else f"  {punto_pred:.3f} m",
            xy=(edad_pred, punto_pred),
            xytext=(edad_pred + 0.8, punto_pred + offset_anot * 1.2),
            fontsize=9, color="#c0392b", fontweight="700",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#e67e22", alpha=0.9)
        )

    ax.set_xlabel("Edad (meses)", fontsize=10, color="#4a4a4a")
    ax.set_ylabel(ylabel, fontsize=10, color="#4a4a4a")
    ax.set_title(titulo, fontsize=11, fontweight="700", color="#1a2e1a", pad=10)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#d4e8d8", loc="upper left")
    ax.grid(True, alpha=0.15, color="#94a3b8", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d4e8d8")
    ax.spines["bottom"].set_color("#d4e8d8")
    xlim_min = -0.3 if (0 in (edades.values if hasattr(edades, "values") else edades)) else 0.5
    ax.set_xlim(xlim_min, 22.5)
    ax.set_ylim(*ylim)
    ax.tick_params(colors="#6b7280", labelsize=8.5)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 1 — CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-pill">📊 Curvas del rancho 2015-2025</div>',
                unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        sx1 = st.radio("Sexo:", ["Machos ♂", "Hembras ♀"], horizontal=True, key="sx1")
    with c2:
        vr1 = st.radio("Variable:", ["Peso (kg)", "Alzada (m)"], horizontal=True, key="vr1")

    sk1 = "M" if "Machos" in sx1 else "H"
    n1  = 111 if sk1 == "M" else 106

    if "Peso" in vr1:
        fig1 = fig_curvas(
            stats_ref[f"stats_{sk1}"], C[sk1],
            f"Curvas de Peso — {'Machos' if sk1=='M' else 'Hembras'} PSI  (n={n1})",
            "Peso (kg)", (20, 570), [0, 6, 12, 18], "{:.0f} kg", 20
        )
    else:
        fig1 = fig_curvas(
            stats_alz[f"stats_{sk1}"], C[sk1],
            f"Curvas de Alzada — {'Machos' if sk1=='M' else 'Hembras'} PSI  (n={n1})",
            "Alzada (m)", (0.85, 1.68), [6, 12, 18], "{:.2f} m", 0.012
        )

    st.pyplot(fig1)
    plt.close(fig1)

    with st.expander("Ver tabla de valores"):
        st_d1 = stats_ref[f"stats_{sk1}"] if "Peso" in vr1 else stats_alz[f"stats_{sk1}"]
        t1 = st_d1[["edad_meses", "p10", "p25", "p50", "p75", "p90", "n"]].copy()
        t1.columns = ["Edad", "P10", "P25", "P50", "P75", "P90", "N"]
        st.dataframe(t1.round(1 if "Peso" in vr1 else 3),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.83rem;color:#2d5a3d;margin-top:0.5rem">
        <strong>Como leer:</strong> La zona sombreada oscura es el rango normal (P25-P75).
        Las lineas punteadas marcan P10 y P90. Un potro fuera de ese rango
        merece seguimiento clinico.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-pill">🔍 Evaluación individual</div>',
                unsafe_allow_html=True)

    col_n, col_s = st.columns([2, 1])
    with col_n:
        nombre2 = st.text_input("Nombre / identificador",
                                placeholder="Ej. Hijo de Mila Race", key="nombre2")
    with col_s:
        sx2 = st.radio("Sexo:", ["Macho ♂", "Hembra ♀"], horizontal=True, key="sx2")
    sk2 = "M" if "Macho" in sx2 else "H"

    st.markdown("""
    <div style="font-size:0.82rem;color:#6b8f72;margin:0.5rem 0 0.75rem">
        Ingresa <strong>peso (kg)</strong> y/o <strong>alzada (m)</strong>
        en los meses disponibles. Minimo 2 meses con peso.
    </div>
    """, unsafe_allow_html=True)

    MESES_CLAVE = [1, 3, 6, 9, 12, 18]
    MESES_EXTRA = [2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22]

    pesos2   = {}
    alzadas2 = {}

    pn_col, _ = st.columns([1, 3])
    with pn_col:
        pnac2 = st.number_input("Peso al nacer (kg) — opcional",
                                min_value=0.0, max_value=80.0,
                                value=0.0, step=0.5, key="pnac2")
    if pnac2 > 0:
        pesos2[0] = pnac2

    cols_k = st.columns(3)
    for i, mes in enumerate(MESES_CLAVE):
        with cols_k[i % 3]:
            st.markdown(f"**Mes {mes}**")
            ca, cb = st.columns(2)
            with ca:
                pv = st.number_input("Peso kg", min_value=0.0, max_value=700.0,
                                     value=0.0, step=1.0, key=f"p2_{mes}")
            with cb:
                av = st.number_input("Alzada m", min_value=0.0, max_value=2.0,
                                     value=0.0, step=0.01, key=f"a2_{mes}")
            if pv > 0:
                pesos2[mes] = pv
            if av > 0:
                alzadas2[mes] = av

    with st.expander("➕ Agregar más meses"):
        cols_e = st.columns(4)
        for i, mes in enumerate(MESES_EXTRA):
            with cols_e[i % 4]:
                st.markdown(f"**Mes {mes}**")
                ca2, cb2 = st.columns(2)
                with ca2:
                    pv2 = st.number_input("kg", min_value=0.0, max_value=700.0,
                                          value=0.0, step=1.0, key=f"p2_{mes}")
                with cb2:
                    av2 = st.number_input("m", min_value=0.0, max_value=2.0,
                                          value=0.0, step=0.01, key=f"a2_{mes}")
                if pv2 > 0:
                    pesos2[mes] = pv2
                if av2 > 0:
                    alzadas2[mes] = av2

    st.markdown("<hr>", unsafe_allow_html=True)
    analizar2 = st.button("🔍 Ver evaluación completa", type="primary",
                          use_container_width=True, key="btn2")

    if analizar2:
        meses_p = sorted([m for m in pesos2 if pesos2[m] > 0 and m > 0])
        if len(meses_p) < 2:
            st.warning("Ingresa al menos 2 mediciones de peso (meses > 0).")
            st.stop()

        sp2 = stats_ref[f"stats_{sk2}"]
        sa2 = stats_alz[f"stats_{sk2}"]
        nombre_d = nombre2 or "Potro evaluado"

        filas2   = []
        alertas2 = []
        alto_cnt = bajo_cnt = 0

        for mes in meses_p:
            ref = sp2[sp2.edad_meses == mes]
            if ref.empty:
                continue
            peso = pesos2[mes]
            p10 = ref["p10"].values[0]
            p25 = ref["p25"].values[0]
            p50 = ref["p50"].values[0]
            p75 = ref["p75"].values[0]
            p90 = ref["p90"].values[0]
            diff = ((peso - p50) / p50) * 100

            if peso < p10:
                zona = "MUY BAJO"; alerta = True;  bajo_cnt += 1
            elif peso < p25:
                zona = "BAJO";     alerta = True;  bajo_cnt += 1
            elif peso <= p75:
                zona = "NORMAL";   alerta = False
            elif peso <= p90:
                zona = "ALTO";     alerta = False; alto_cnt += 1
            else:
                zona = "MUY ALTO"; alerta = True;  alto_cnt += 1

            if alerta:
                alertas2.append(mes)

            alz_zona = None
            if mes in alzadas2 and alzadas2[mes] > 0:
                rfa = sa2[sa2.edad_meses == mes]
                if not rfa.empty:
                    if alzadas2[mes] < rfa["p25"].values[0]:
                        alz_zona = "Baja"
                    elif alzadas2[mes] <= rfa["p75"].values[0]:
                        alz_zona = "Normal"
                    else:
                        alz_zona = "Alta"

            filas2.append({
                "mes": mes, "peso": peso,
                "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
                "diff": diff, "zona": zona, "alerta": alerta,
                "alzada": alzadas2.get(mes, None), "alz_zona": alz_zona
            })

        n_f      = len(filas2)
        pct_alto = alto_cnt / n_f if n_f > 0 else 0
        pct_bajo = bajo_cnt / n_f if n_f > 0 else 0
        vals_p   = [f["peso"] for f in filas2]
        perdidas = sum(1 for i in range(1, len(vals_p)) if vals_p[i] < vals_p[i - 1])
        caida    = any((vals_p[i] - vals_p[i - 1]) / vals_p[i - 1] * 100 < -8
                       for i in range(1, len(vals_p)))

        if (perdidas >= 4) or caida:
            patron2 = "Patrón Irregular"; cls2 = "p-irregular"
            desc2   = "Pérdidas de peso detectadas entre mediciones consecutivas."
            inds = [
                ("alert", "Evaluación clinica urgente — perdida de peso en potro en crecimiento."),
                ("alert", "Descartar enfermedades gastrointestinales, parasitosis o estres."),
                ("warn",  "Revisar calidad y cantidad del alimento ofrecido."),
                ("warn",  "Verificar acceso a agua limpia y comedero."),
            ]
        elif pct_alto >= 0.60:
            patron2 = "Patrón Superior"; cls2 = "p-superior"
            desc2   = "Crecimiento consistentemente por encima del promedio del rancho."
            inds = [
                ("info", "Crecimiento excelente — por encima del P75 en la mayoria de meses."),
                ("",     "Mantener el plan nutricional y de manejo actual."),
                ("warn", "Vigilar condicion corporal para evitar sobrepeso tardio."),
            ]
        elif pct_bajo >= 0.60:
            patron2 = "Patrón Inferior"; cls2 = "p-inferior"
            desc2   = "Crecimiento persistentemente por debajo del rango esperado."
            inds = [
                ("warn", "Revisar aporte energetico — considerar incrementar concentrado."),
                ("warn", "Evaluar estado de desparasitacion — alta carga reduce absorcion."),
                ("warn", "Verificar disponibilidad de forraje y agua limpia."),
                ("",     "Repetir evaluacion en 4 semanas tras ajuste nutricional."),
            ]
        else:
            patron2 = "Patrón Normal"; cls2 = "p-normal"
            desc2   = "Crecimiento dentro del rango esperado para el rancho."
            inds = [
                ("", "Mantener el programa de manejo y alimentacion actual."),
                ("", "Continuar con pesajes mensuales para seguimiento."),
            ]
            if alertas2:
                inds.append(("warn", f"Vigilar meses con alerta: {alertas2}."))

        st.markdown("<hr>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f'<div class="patron-box {cls2}">{patron2}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.83rem;color:#4a6a4a;margin-bottom:0.75rem">{desc2}</div>',
                        unsafe_allow_html=True)

            ganancia = round(vals_p[-1] - vals_p[0]) if len(vals_p) >= 2 else 0
            norm_pct = round(sum(1 for f in filas2 if f["zona"] == "NORMAL") / n_f * 100)
            cls_n    = "ok" if norm_pct >= 60 else ("warn" if norm_pct >= 40 else "bad")
            cls_a    = "" if len(alertas2) == 0 else ("warn" if len(alertas2) <= 2 else "bad")

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card {cls_n}">
                <div class="sv">{norm_pct}%</div><div class="sl">En rango normal</div>
              </div>
              <div class="stat-card">
                <div class="sv">+{ganancia} kg</div><div class="sl">Ganancia total</div>
              </div>
              <div class="stat-card {cls_a}">
                <div class="sv">{len(alertas2)}</div><div class="sl">Alertas</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if len(alertas2) == 0:
                st.markdown('<div class="ok-strip">✅ Sin alertas — crecimiento dentro del rango</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-strip">⚠️ Alertas en meses: {alertas2}</div>',
                            unsafe_allow_html=True)

            ind_html = "".join([
                f'<div class="ind-item ind-{c if c else "ok"}">'
                f'<div class="ind-dot"></div><span>{t}</span></div>'
                for c, t in inds
            ])
            st.markdown(
                f'<div class="ind-box"><div class="ind-titulo">Indicaciones clinicas</div>'
                f'{ind_html}</div>',
                unsafe_allow_html=True
            )

        with r2:
            pesos_dict = {f["mes"]: f["peso"] for f in filas2}
            ylim_p     = (50, max(max(pesos_dict.values()) + 60, 520))
            fig_p = fig_curvas(
                sp2, C[sk2],
                f"Peso — {nombre_d} vs. Curvas de referencia",
                "Peso (kg)", ylim_p,
                [6, 12, 18], "{:.0f} kg", 20,
                datos_potro=pesos_dict, nombre_potro=nombre_d
            )
            st.pyplot(fig_p)
            plt.close(fig_p)

            alz_dict = {m: alzadas2[m] for m in alzadas2 if alzadas2[m] > 0 and m > 0}
            if len(alz_dict) >= 2:
                fig_a = fig_curvas(
                    sa2, C[sk2],
                    f"Alzada — {nombre_d} vs. Curvas de referencia",
                    "Alzada (m)", (0.85, 1.70),
                    [6, 12, 18], "{:.2f} m", 0.012,
                    datos_potro=alz_dict, nombre_potro=nombre_d
                )
                st.pyplot(fig_a)
                plt.close(fig_a)

        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("Ver evaluación detallada mes a mes"):
            df2 = pd.DataFrame([{
                "Mes":           f["mes"],
                "Peso real (kg)":f["peso"],
                "P25":           round(f["p25"], 1),
                "P50":           round(f["p50"], 1),
                "P75":           round(f["p75"], 1),
                "vs P50":        f'{f["diff"]:+.1f}%',
                "Estado peso":   f["zona"],
                "Alzada (m)":    f["alzada"] if f["alzada"] else "—",
                "Estado alzada": f["alz_zona"] if f["alz_zona"] else "—",
            } for f in filas2])

            def color_estado(v):
                if "BAJO" in str(v):
                    return "background:#fff3e0;color:#92400e"
                elif "ALTO" in str(v):
                    return "background:#dbeafe;color:#1e40af"
                elif "NORMAL" in str(v):
                    return "background:#e8f4ee;color:#1a4731"
                return ""

            try:
                styled = df2.style.map(color_estado, subset=["Estado peso"])
            except AttributeError:
                styled = df2.style.applymap(color_estado, subset=["Estado peso"])

            st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-pill">🎯 Predictor de peso y alzada</div>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([1, 1, 1])
    with pc1:
        sx3 = st.radio("Sexo:", ["Macho ♂", "Hembra ♀"], horizontal=True, key="sx3")
    with pc2:
        edad3 = st.slider("Edad (meses):", 1, 22, 6, key="edad3")
    with pc3:
        alz3 = st.number_input("Alzada actual (m) — opcional",
                               min_value=0.0, max_value=2.0,
                               value=0.0, step=0.01, key="alz3",
                               help="Mejora la precision: R2=0.9641 vs 0.9458")

    sk3   = "M" if "Macho" in sx3 else "H"
    sbin3 = 1 if sk3 == "M" else 0
    sp3   = stats_ref[f"stats_{sk3}"]
    sa3   = stats_alz[f"stats_{sk3}"]
    ref_a3 = sa3[sa3.edad_meses == edad3]
    alz_m3 = ref_a3["p50"].values[0] if len(ref_a3) > 0 else 1.35
    alz_u3 = alz3 if alz3 > 0 else alz_m3

    peso3  = mod_peso.predict([[sbin3, edad3, alz_u3]])[0]
    alzp3  = mod_alz.predict([[sbin3, edad3, peso3]])[0]
    ref_p3 = sp3[sp3.edad_meses == edad3]

    st.markdown("<hr>", unsafe_allow_html=True)

    if alz3 > 0:
        st.markdown('<div class="ok-strip">Usando modelo mejorado con alzada — R2=0.9641</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#f0f9ff;border:0.5px solid #bae6fd;'
            'border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#0c4a6e;'
            'margin-bottom:0.5rem">Ingresa la alzada para mayor precision</div>',
            unsafe_allow_html=True
        )

    if len(ref_p3) > 0:
        p25r = ref_p3["p25"].values[0]
        p50r = ref_p3["p50"].values[0]
        p75r = ref_p3["p75"].values[0]

        if peso3 < p25r:
            pos = "Inferior al rango normal"; pc_col = "#92400e"; pb = "#fff3e0"
        elif peso3 <= p75r:
            pos = "Dentro del rango normal";  pc_col = "#1a4731"; pb = "#e8f4ee"
        else:
            pos = "Superior al rango normal"; pc_col = "#1e40af"; pb = "#dbeafe"

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(f"""
            <div style="background:{pb};border:1.5px solid {pc_col};
                        border-radius:10px;padding:1rem;text-align:center">
              <div style="font-size:2rem;font-weight:700;color:{pc_col}">{peso3:.0f} kg</div>
              <div style="font-size:0.78rem;color:{pc_col};opacity:0.85;margin-top:3px">
                Peso predicho · mes {edad3}
              </div>
              <div style="font-size:0.72rem;color:{pc_col};opacity:0.7;margin-top:4px;
                          background:rgba(255,255,255,0.5);border-radius:6px;padding:2px 6px">
                {pos}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.metric("P25 rancho", f"{p25r:.0f} kg")
        with rc3:
            st.metric("P50 rancho", f"{p50r:.0f} kg")
        with rc4:
            st.metric("P75 rancho", f"{p75r:.0f} kg")

        st.markdown("")
        if len(ref_a3) > 0:
            a25 = ref_a3["p25"].values[0]
            a50 = ref_a3["p50"].values[0]
            a75 = ref_a3["p75"].values[0]
            ra1, ra2, ra3, ra4 = st.columns(4)
            with ra1:
                st.metric("Alzada predicha", f"{alzp3:.3f} m")
            with ra2:
                st.metric("P25 alzada", f"{a25:.3f} m")
            with ra3:
                st.metric("P50 alzada", f"{a50:.3f} m")
            with ra4:
                st.metric("P75 alzada", f"{a75:.3f} m")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Curvas de predicción — punto marcado en la edad seleccionada**")

    edades_g = list(range(1, 23))
    saM3 = stats_alz["stats_M"]
    saH3 = stats_alz["stats_H"]
    spM3 = stats_ref["stats_M"]
    spH3 = stats_ref["stats_H"]

    aM3 = [saM3[saM3.edad_meses == e]["p50"].values[0]
           if len(saM3[saM3.edad_meses == e]) > 0 else 1.35 for e in edades_g]
    aH3 = [saH3[saH3.edad_meses == e]["p50"].values[0]
           if len(saH3[saH3.edad_meses == e]) > 0 else 1.33 for e in edades_g]
    pM3 = [mod_peso.predict([[1, e, a]])[0] for e, a in zip(edades_g, aM3)]
    pH3 = [mod_peso.predict([[0, e, a]])[0] for e, a in zip(edades_g, aH3)]

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor("#f8faf8")
    ax3.set_facecolor("#f8faf8")
    ax3.fill_between(spM3.edad_meses, spM3.p25, spM3.p75,
                     alpha=0.12, color=C["M"], label="Rango normal Machos")
    ax3.fill_between(spH3.edad_meses, spH3.p25, spH3.p75,
                     alpha=0.12, color=C["H"], label="Rango normal Hembras")
    ax3.plot(edades_g, pM3, color=C["M"], linewidth=2.2, label="Prediccion Machos")
    ax3.plot(edades_g, pH3, color=C["H"], linewidth=2.2, label="Prediccion Hembras")
    ax3.axvline(x=edad3, color="#94a3b8", linestyle="--", linewidth=1.5,
                alpha=0.8, label=f"Mes {edad3}")
    ax3.scatter([edad3], [peso3], color="#e67e22", s=200, zorder=7,
                edgecolors="white", linewidths=2.5,
                label=f"Prediccion: {peso3:.0f} kg")
    ax3.annotate(
        f"  {peso3:.0f} kg",
        xy=(edad3, peso3), xytext=(edad3 + 0.8, peso3 + 20),
        fontsize=10, color="#c0392b", fontweight="700",
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff3e0", ec="#e67e22", alpha=0.95)
    )
    ax3.set_xlabel("Edad (meses)", fontsize=11)
    ax3.set_ylabel("Peso (kg)", fontsize=11)
    ax3.set_title("Predicciones vs. rangos normales del rancho",
                  fontsize=12, fontweight="700")
    ax3.legend(fontsize=9, framealpha=0.9, edgecolor="#d4e8d8")
    ax3.grid(True, alpha=0.15, linestyle="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_xlim(0.5, 22.5)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    with st.expander("Ver tabla completa"):
        st.dataframe(pd.DataFrame({
            "Mes":               edades_g,
            "Peso Machos(kg)":   [round(p) for p in pM3],
            "Peso Hembras(kg)":  [round(p) for p in pH3],
            "Alzada Machos(m)":  [round(a, 3) for a in aM3],
            "Alzada Hembras(m)": [round(a, 3) for a in aH3],
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIÓN INTERNACIONAL
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-pill">🌍 Rancho MX vs Literatura Internacional</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.9rem 1.1rem;font-size:0.85rem;color:#2d5a3d;margin-bottom:1rem">
        Comparacion del P50 del rancho mexicano contra datos publicados en
        tres estudios internacionales de referencia en PSI.
        Los valores de literatura corresponden a medias poblacionales.
    </div>
    """, unsafe_allow_html=True)

    # Datos de literatura
    hintz_m  = {0:55,1:98,2:132,3:170,4:195,5:221,6:245,7:270,
                8:283,9:310,10:318,11:334,12:345,13:359,14:373,15:392,16:415,17:428,18:446}
    hintz_h  = {0:54,1:97,2:131,3:166,4:192,5:212,6:236,7:260,
                8:272,9:296,10:304,11:320,12:329,13:343,14:355,15:375,16:392,17:406,18:424}
    hintz_am = {0:100.6,1:110.8,2:118.5,3:125.2,4:128.9,5:131.6,6:134.6,
                7:137.1,8:139.5,9:141.8,10:142.6,11:144.4,12:145.9,13:147.2,
                14:148.8,15:150.2,16:151.8,17:152.8,18:154.5}

    ker_meses  = [0,    1,    6,    12,   18]
    ker_ky     = [67.5, 99.3, 250.7,353.3,453.9]
    ker_world  = [66.9, 98.6, 247.1,350.7,444.9]
    ker_aus    = [69.6, 102.4,251.4,357.8,460.7]
    ker_alz_ky = [105.7,112.6,135.9,147.8,154.7]
    ker_alz_w  = [106.1,112.0,135.0,147.1,153.8]

    dc_meses = [0,    6,    12,   18]
    dc_m     = [53.8, 243.8,337.2,432.7]
    dc_h     = [56.6, 248.2,343.6,445.5]
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

    COL_COMP = {
        "MX":    "#1a4731",
        "Hintz": "#e74c3c",
        "KY":    "#2980b9",
        "World": "#8e44ad",
        "Aus":   "#16a085",
        "BR":    "#f39c12",
    }

    var_comp  = st.radio("Variable a comparar:", ["Peso (kg)", "Alzada (cm)"],
                         horizontal=True, key="var_comp")
    sexo_comp = st.radio("Sexo:", ["Machos", "Hembras"],
                         horizontal=True, key="sexo_comp")

    fig_c, ax_c = plt.subplots(figsize=(13, 5.5))
    fig_c.patch.set_facecolor("#f8faf8")
    ax_c.set_facecolor("#f8faf8")

    if "Peso" in var_comp:
        if sexo_comp == "Machos":
            ax_c.plot(list(mx_m_p50.keys()), list(mx_m_p50.values()),
                      color=COL_COMP["MX"], linewidth=3.5, marker="o",
                      markersize=5, label="Rancho MX — P50 (n=111)", zorder=6)
            ax_c.plot(list(hintz_m.keys()), list(hintz_m.values()),
                      color=COL_COMP["Hintz"], linewidth=2, linestyle="--",
                      alpha=0.85, label="Hintz et al. 1979 — Canada")
            ax_c.plot(ker_meses, ker_ky, color=COL_COMP["KY"], linewidth=2,
                      linestyle="-.", marker="s", markersize=7,
                      label="KER — Kentucky (Pagan 2009)")
            ax_c.plot(ker_meses, ker_aus, color=COL_COMP["Aus"], linewidth=1.5,
                      linestyle=":", marker="^", markersize=6,
                      label="KER — Australia (Pagan 2009)")
            ax_c.plot(ker_meses, ker_world, color=COL_COMP["World"], linewidth=2,
                      linestyle="-.", alpha=0.8, label="KER — Promedio mundial")
            ax_c.plot(dc_meses, dc_m, color=COL_COMP["BR"], linewidth=2,
                      marker="D", markersize=8, label="De Castro et al. 2021 — Brasil")
        else:
            ax_c.plot(list(mx_h_p50.keys()), list(mx_h_p50.values()),
                      color=COL_COMP["MX"], linewidth=3.5, marker="o",
                      markersize=5, label="Rancho MX — P50 (n=106)", zorder=6)
            ax_c.plot(list(hintz_h.keys()), list(hintz_h.values()),
                      color=COL_COMP["Hintz"], linewidth=2, linestyle="--",
                      alpha=0.85, label="Hintz et al. 1979 — Canada")
            ax_c.plot(ker_meses, ker_ky, color=COL_COMP["KY"], linewidth=2,
                      linestyle="-.", marker="s", markersize=7,
                      label="KER — Kentucky (Pagan 2009)")
            ax_c.plot(ker_meses, ker_world, color=COL_COMP["World"], linewidth=2,
                      linestyle="-.", alpha=0.8, label="KER — Promedio mundial")
            ax_c.plot(dc_meses, dc_h, color=COL_COMP["BR"], linewidth=2,
                      marker="D", markersize=8, label="De Castro et al. 2021 — Brasil")
        ax_c.set_ylabel("Peso (kg)", fontsize=11)
        titulo_var = "Peso corporal"
    else:
        if sexo_comp == "Machos":
            ax_c.plot(list(mx_am_p50.keys()), list(mx_am_p50.values()),
                      color=COL_COMP["MX"], linewidth=3.5, marker="o",
                      markersize=5, label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_am.keys()), list(hintz_am.values()),
                      color=COL_COMP["Hintz"], linewidth=2, linestyle="--",
                      alpha=0.85, label="Hintz et al. 1979 — Canada")
            ax_c.plot(ker_meses, ker_alz_ky, color=COL_COMP["KY"], linewidth=2,
                      linestyle="-.", marker="s", markersize=7,
                      label="KER — Kentucky (Pagan 2009)")
            ax_c.plot(ker_meses, ker_alz_w, color=COL_COMP["World"], linewidth=2,
                      linestyle="-.", alpha=0.8, label="KER — Promedio mundial")
            ax_c.plot(dc_meses, dc_alz_m, color=COL_COMP["BR"], linewidth=2,
                      marker="D", markersize=8, label="De Castro et al. 2021 — Brasil")
        else:
            ax_c.plot(list(mx_ah_p50.keys()), list(mx_ah_p50.values()),
                      color=COL_COMP["MX"], linewidth=3.5, marker="o",
                      markersize=5, label="Rancho MX — P50", zorder=6)
            ax_c.plot(list(hintz_am.keys()), list(hintz_am.values()),
                      color=COL_COMP["Hintz"], linewidth=2, linestyle="--",
                      alpha=0.85, label="Hintz et al. 1979 — Canada (machos)")
            ax_c.plot(ker_meses, ker_alz_ky, color=COL_COMP["KY"], linewidth=2,
                      linestyle="-.", marker="s", markersize=7,
                      label="KER — Kentucky (Pagan 2009)")
            ax_c.plot(ker_meses, ker_alz_w, color=COL_COMP["World"], linewidth=2,
                      linestyle="-.", alpha=0.8, label="KER — Promedio mundial")
            ax_c.plot(dc_meses, dc_alz_h, color=COL_COMP["BR"], linewidth=2,
                      marker="D", markersize=8, label="De Castro et al. 2021 — Brasil")
        ax_c.set_ylabel("Alzada (cm)", fontsize=11)
        titulo_var = "Alzada a la cruz"

    ax_c.set_xlabel("Edad (meses)", fontsize=11)
    ax_c.set_title(
        f"{titulo_var} — {sexo_comp} PSI\n"
        f"Rancho mexicano 2015-2025 vs literatura internacional",
        fontsize=12, fontweight="700"
    )
    ax_c.legend(fontsize=9, framealpha=0.9, edgecolor="#d4e8d8")
    ax_c.grid(True, alpha=0.15, linestyle="--")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_xlim(-0.5, 22)
    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)

    # Tabla comparativa
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Tabla comparativa en puntos clave")

    meses_k = [0, 6, 12, 18]

    def ker_val(lista, meses_lista, meses_target):
        d = dict(zip(meses_lista, lista))
        return [round(d[m], 1) if m in d else "—" for m in meses_target]

    if "Peso" in var_comp and sexo_comp == "Machos":
        data_tab = {
            "Edad (meses)":      meses_k,
            "Rancho MX (kg)":    [mx_m_p50.get(m, "—")  for m in meses_k],
            "Hintz 1979 Canada": [hintz_m.get(m, "—")   for m in meses_k],
            "KER Kentucky":      ker_val(ker_ky,    ker_meses, meses_k),
            "KER Mundial":       ker_val(ker_world,  ker_meses, meses_k),
            "Brasil 2021":       ker_val(dc_m,       dc_meses,  meses_k),
        }
    elif "Peso" in var_comp and sexo_comp == "Hembras":
        data_tab = {
            "Edad (meses)":      meses_k,
            "Rancho MX (kg)":    [mx_h_p50.get(m, "—")  for m in meses_k],
            "Hintz 1979 Canada": [hintz_h.get(m, "—")   for m in meses_k],
            "KER Kentucky":      ker_val(ker_ky,    ker_meses, meses_k),
            "KER Mundial":       ker_val(ker_world,  ker_meses, meses_k),
            "Brasil 2021":       ker_val(dc_h,       dc_meses,  meses_k),
        }
    elif "Alzada" in var_comp and sexo_comp == "Machos":
        data_tab = {
            "Edad (meses)":      meses_k,
            "Rancho MX (cm)":    [mx_am_p50.get(m, "—") for m in meses_k],
            "Hintz 1979 Canada": [hintz_am.get(m, "—")  for m in meses_k],
            "KER Kentucky":      ker_val(ker_alz_ky, ker_meses, meses_k),
            "KER Mundial":       ker_val(ker_alz_w,  ker_meses, meses_k),
            "Brasil 2021":       ker_val(dc_alz_m,   dc_meses,  meses_k),
        }
    else:
        data_tab = {
            "Edad (meses)":      meses_k,
            "Rancho MX (cm)":    [mx_ah_p50.get(m, "—") for m in meses_k],
            "Hintz 1979 Canada": [hintz_am.get(m, "—")  for m in meses_k],
            "KER Kentucky":      ker_val(ker_alz_ky, ker_meses, meses_k),
            "KER Mundial":       ker_val(ker_alz_w,  ker_meses, meses_k),
            "Brasil 2021":       ker_val(dc_alz_h,   dc_meses,  meses_k),
        }

    st.dataframe(pd.DataFrame(data_tab), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:1rem 1.2rem;font-size:0.85rem;color:#2d5a3d">
        <strong>Interpretacion de resultados:</strong><br><br>
        Al <strong>nacimiento</strong>, el rancho mexicano muestra pesos similares a
        Canada (Hintz, 1979) y Brasil (De Castro, 2021), pero inferiores a Kentucky y
        el promedio mundial KER. Esto es consistente con diferencias geneticas y de
        seleccion por industria hipica (Pagan, 2025).<br><br>
        A los <strong>6 meses</strong>, las diferencias se reducen a menos de 5 kg en
        todas las poblaciones, sugiriendo que el manejo nutricional postnatal del rancho
        es comparable al estandar internacional.<br><br>
        A los <strong>18 meses</strong>, el rancho mexicano muestra una brecha de 17-26 kg
        respecto a las referencias angloamericanas, atribuible a que los animales son
        vendidos al hipodromo antes de completar el ciclo de preparacion de yearling.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — REPORTE IA
# ══════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-pill">🤖 Reporte Clínico con IA</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.9rem 1.1rem;font-size:0.85rem;color:#2d5a3d;margin-bottom:1rem">
        Ingresa los datos del potro y el sistema generará automáticamente un
        reporte clínico narrativo con interpretación y recomendaciones,
        utilizando inteligencia artificial (Claude Sonnet).
    </div>
    """, unsafe_allow_html=True)

    # ── Entrada de datos ──
    st.subheader("Datos del potro")

    col_ia1, col_ia2, col_ia3 = st.columns(3)
    with col_ia1:
        nombre_ia = st.text_input("Nombre / identificador",
                                  placeholder="Ej. Hijo de Mila Race",
                                  key="nombre_ia")
    with col_ia2:
        sexo_ia = st.radio("Sexo:", ["Macho", "Hembra"],
                           horizontal=True, key="sexo_ia")
    with col_ia3:
        rancho_ia = st.text_input("Rancho", value="Rancho PSI México",
                                  key="rancho_ia")

    st.markdown("**Mediciones disponibles** — ingresa los datos que tengas:")

    MESES_IA = [0, 1, 3, 6, 9, 12, 18]
    pesos_ia = {}
    alzadas_ia = {}

    cols_ia = st.columns(4)
    for i, mes in enumerate(MESES_IA):
        with cols_ia[i % 4]:
            lbl = "Nacimiento" if mes == 0 else f"Mes {mes}"
            st.markdown(f"**{lbl}**")
            c1, c2 = st.columns(2)
            with c1:
                pv = st.number_input("Peso kg", min_value=0.0, max_value=700.0,
                                     value=0.0, step=1.0, key=f"pia_{mes}")
            with c2:
                av = st.number_input("Alzada m", min_value=0.0, max_value=2.0,
                                     value=0.0, step=0.01, key=f"aia_{mes}")
            if pv > 0:
                pesos_ia[mes] = pv
            if av > 0:
                alzadas_ia[mes] = av

    # Contexto clínico adicional
    st.markdown("**Contexto clínico adicional** — opcional")
    col_ctx1, col_ctx2 = st.columns(2)
    with col_ctx1:
        antecedentes = st.text_area(
            "Antecedentes (enfermedades, tratamientos, etc.)",
            placeholder="Ej. Desparasitado a los 2 meses, sin enfermedades reportadas...",
            height=80, key="antecedentes_ia"
        )
    with col_ctx2:
        manejo = st.text_area(
            "Manejo y alimentación",
            placeholder="Ej. Pastoreo libre + concentrado 1kg/día desde mes 3...",
            height=80, key="manejo_ia"
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    generar_ia = st.button("🤖 Generar reporte clínico con IA",
                           type="primary", use_container_width=True,
                           key="btn_ia")

    if generar_ia:
        meses_con_peso = [m for m in pesos_ia if pesos_ia[m] > 0]
        if len(meses_con_peso) < 2:
            st.warning("Ingresa al menos 2 mediciones de peso para generar el reporte.")
            st.stop()

        # ── Calcular percentiles para cada medición ──
        sk_ia = "M" if sexo_ia == "Macho" else "H"
        sp_ia = stats_ref[f"stats_{sk_ia}"]
        sa_ia = stats_alz[f"stats_{sk_ia}"]

        datos_eval = []
        for mes in sorted(meses_con_peso):
            if mes == 0:
                datos_eval.append({
                    "mes": mes,
                    "etiqueta": "Nacimiento",
                    "peso": pesos_ia[mes],
                    "alzada": alzadas_ia.get(mes),
                    "percentil_peso": "Sin referencia (mes 0)",
                    "zona_peso": "N/A"
                })
                continue

            ref = sp_ia[sp_ia.edad_meses == mes]
            if ref.empty:
                continue

            peso = pesos_ia[mes]
            p10  = ref["p10"].values[0]
            p25  = ref["p25"].values[0]
            p50  = ref["p50"].values[0]
            p75  = ref["p75"].values[0]
            p90  = ref["p90"].values[0]
            diff = ((peso - p50) / p50) * 100

            if peso < p10:
                zona = "MUY BAJO (< P10)"
            elif peso < p25:
                zona = "BAJO (P10-P25)"
            elif peso <= p75:
                zona = "NORMAL (P25-P75)"
            elif peso <= p90:
                zona = "ALTO (P75-P90)"
            else:
                zona = "MUY ALTO (> P90)"

            alz_info = ""
            if mes in alzadas_ia and alzadas_ia[mes] > 0:
                rfa = sa_ia[sa_ia.edad_meses == mes]
                if not rfa.empty:
                    a50 = rfa["p50"].values[0]
                    da  = ((alzadas_ia[mes] - a50) / a50) * 100
                    alz_info = f"{alzadas_ia[mes]:.2f} m ({da:+.1f}% vs mediana {a50:.2f} m)"

            datos_eval.append({
                "mes":            mes,
                "etiqueta":       f"Mes {mes}",
                "peso":           peso,
                "p25":            round(p25, 1),
                "p50":            round(p50, 1),
                "p75":            round(p75, 1),
                "diff_pct":       round(diff, 1),
                "zona_peso":      zona,
                "alzada_info":    alz_info,
            })

        # ── Clasificar patrón ──
        vals_p = [d["peso"] for d in datos_eval if d["mes"] > 0]
        n_bajo = sum(1 for d in datos_eval if "BAJO" in d.get("zona_peso", ""))
        n_alto = sum(1 for d in datos_eval if "ALTO" in d.get("zona_peso", ""))
        n_eval = len([d for d in datos_eval if d["mes"] > 0])

        perdidas = sum(1 for i in range(1, len(vals_p)) if vals_p[i] < vals_p[i - 1])
        caida    = any((vals_p[i] - vals_p[i - 1]) / vals_p[i - 1] * 100 < -8
                       for i in range(1, len(vals_p)))

        if (perdidas >= 4) or caida:
            patron_ia = "Irregular"
        elif n_alto / n_eval >= 0.6 if n_eval > 0 else False:
            patron_ia = "Superior"
        elif n_bajo / n_eval >= 0.6 if n_eval > 0 else False:
            patron_ia = "Inferior"
        else:
            patron_ia = "Normal"

        # ── Construir prompt para Claude ──
        tabla_mediciones = "\n".join([
            f"  - {d['etiqueta']}: {d['peso']} kg "
            f"| Percentil: {d.get('zona_peso','N/A')} "
            f"| {d['diff_pct']:+.1f}% vs mediana "
            f"{'| Alzada: ' + d['alzada_info'] if d.get('alzada_info') else ''}"
            if d["mes"] > 0 else
            f"  - {d['etiqueta']}: {d['peso']} kg (peso al nacer)"
            for d in datos_eval
        ])

        prompt = f"""Eres un médico veterinario especialista en equinos con experiencia en cría de Pura Sangre Inglés (PSI).

Debes generar un REPORTE CLÍNICO PROFESIONAL sobre el crecimiento de un potro PSI basándote en los datos de monitoreo mensual, comparados contra curvas de referencia percentiladas construidas con 217 potros PSI de un rancho mexicano (2015-2025).

DATOS DEL POTRO:
- Nombre: {nombre_ia or 'Sin nombre'}
- Sexo: {sexo_ia}
- Rancho: {rancho_ia}
- Patrón de crecimiento clasificado: {patron_ia}

MEDICIONES Y PERCENTILES (referencia: rancho PSI México, P25-P75 = rango normal):
{tabla_mediciones}

ANTECEDENTES CLÍNICOS: {antecedentes if antecedentes else 'No especificados'}
MANEJO Y ALIMENTACIÓN: {manejo if manejo else 'No especificado'}

GENERA UN REPORTE CLÍNICO que incluya:

1. **RESUMEN EJECUTIVO** (2-3 oraciones con el hallazgo principal)

2. **EVALUACIÓN DEL CRECIMIENTO**
   - Análisis del patrón de crecimiento identificado
   - Comparación con la población de referencia del rancho
   - Tendencia general (progresión, estabilidad, irregularidades)

3. **HALLAZGOS RELEVANTES**
   - Meses con valores fuera del rango normal (si los hay)
   - Correlación entre peso y alzada (si se tienen ambos datos)
   - Velocidad de ganancia de peso entre mediciones

4. **RECOMENDACIONES CLÍNICAS**
   - Al menos 3 recomendaciones específicas y accionables
   - Ordenadas de mayor a menor urgencia
   - Basadas en el patrón de crecimiento identificado

5. **SEGUIMIENTO SUGERIDO**
   - Frecuencia de monitoreo recomendada
   - Indicadores de alarma que requieren atención veterinaria inmediata
   - Próxima evaluación sugerida

El reporte debe ser profesional, en español, con terminología veterinaria apropiada pero comprensible para el personal del rancho. Usa formato con secciones claramente delimitadas."""

        # ── Llamar a la Gemini──
        with st.spinner("Generando reporte clínico con IA..."):
            try:
                import requests
                import os

                api_key = os.environ.get("GEMINI_API_KEY", "")
                if not api_key:
                    st.error("API key no configurada. Agrega GEMINI_API_KEY en los secretos de Streamlit.")
                    st.stop()

                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash:generateContent?key={api_key}"
                )

                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 1500,
                    }
                }

                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )

                data = response.json()

                # Extraer texto de la respuesta de Gemini
                if (
                    "candidates" in data
                    and len(data["candidates"]) > 0
                    and "content" in data["candidates"][0]
                    and "parts" in data["candidates"][0]["content"]
                ):
                    reporte_texto = data["candidates"][0]["content"]["parts"][0]["text"]

                    # ── Mostrar reporte ──
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:#1a4731;color:white;border-radius:10px;
                                padding:1rem 1.5rem;margin-bottom:1rem">
                        <div style="font-size:1.1rem;font-weight:700">
                            Reporte Clínico — {nombre_ia or 'Potro evaluado'}
                        </div>
                        <div style="font-size:0.8rem;opacity:0.8;margin-top:0.3rem">
                            {sexo_ia} · {rancho_ia} · Patrón: {patron_ia} · 
                            Generado con Gemini 1.5 Flash (Google AI)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(reporte_texto)

                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.download_button(
                        label="📄 Descargar reporte en .txt",
                        data=(
                            f"REPORTE CLÍNICO CRECIPSI\n"
                            f"{'='*50}\n"
                            f"Potro: {nombre_ia or 'Sin nombre'}\n"
                            f"Sexo: {sexo_ia}\n"
                            f"Rancho: {rancho_ia}\n"
                            f"Patrón: {patron_ia}\n"
                            f"Modelo: Gemini 1.5 Flash (Google AI)\n"
                            f"{'='*50}\n\n"
                            f"{reporte_texto}"
                        ),
                        file_name=f"reporte_{(nombre_ia or 'potro').replace(' ','_')}.txt",
                        mime="text/plain"
                    )

                elif "error" in data:
                    msg = data["error"].get("message", "Error desconocido")
                    st.error(f"Error de Gemini: {msg}")
                else:
                    st.error("No se recibió respuesta del modelo. Intenta de nuevo.")
                    st.json(data)

            except Exception as e:
                st.error(f"Error al conectar con Gemini: {str(e)}")
                if "content" in data and len(data["content"]) > 0:
                    reporte_texto = data["content"][0]["text"]

                    # ── Mostrar reporte ──
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:#1a4731;color:white;border-radius:10px;
                                padding:1rem 1.5rem;margin-bottom:1rem">
                        <div style="font-size:1.1rem;font-weight:700">
                            Reporte Clínico — {nombre_ia or 'Potro evaluado'}
                        </div>
                        <div style="font-size:0.8rem;opacity:0.8;margin-top:0.3rem">
                            {sexo_ia} · {rancho_ia} · Patrón: {patron_ia} · 
                            Generado con IA (Claude Sonnet)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(reporte_texto)

                    # Botón para copiar
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.download_button(
                        label="📄 Descargar reporte en .txt",
                        data=f"REPORTE CLÍNICO CRECIPSI\n"
                             f"{'='*50}\n"
                             f"Potro: {nombre_ia or 'Sin nombre'}\n"
                             f"Sexo: {sexo_ia}\n"
                             f"Rancho: {rancho_ia}\n"
                             f"Patrón: {patron_ia}\n"
                             f"{'='*50}\n\n"
                             f"{reporte_texto}",
                        file_name=f"reporte_{(nombre_ia or 'potro').replace(' ','_')}.txt",
                        mime="text/plain"
                    )

                elif "error" in data:
                    st.error(f"Error de API: {data['error'].get('message', 'Error desconocido')}")
                else:
                    st.error("No se recibió respuesta del modelo. Intenta de nuevo.")

            except Exception as e:
                st.error(f"Error al conectar con la API: {str(e)}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Referencias (Vancouver)")
    st.markdown("""
    1. Hintz HF et al. Growth rate of thoroughbreds. *J Anim Sci.* 1979;48(3):480-487.
    2. Brown-Douglas CG, Pagan JD. Body weight, wither height and growth rates in Thoroughbreds. *Adv Eq Nutr IV.* 2009:213-220.
    3. De Castro LL et al. Body development from birth to 18 months of Thoroughbred foals in Brazil. *Int J Plant Anim Environ Sci.* 2021;11(3):352-362.
    4. National Research Council. *Nutrient Requirements of Horses.* 6th ed. NAP; 2007.
    5. WHO MGRS Group. *WHO child growth standards.* WHO; 2006.
    6. James G et al. *An Introduction to Statistical Learning.* 2nd ed. Springer; 2021.
    7. Dohoo I et al. *Veterinary Epidemiologic Research.* 2nd ed. VER Inc; 2009.
    """)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.83rem;color:#2d5a3d;margin-top:0.5rem">
        <strong>Citacion sugerida:</strong> [Autor]. CreciPSI: Sistema de monitoreo
        inteligente de crecimiento en potros Pura Sangre Ingles mediante inteligencia
        artificial. FMVZ-UNAM. Diplomado en IA en Salud Global. 2026.
    </div>
    """, unsafe_allow_html=True)
