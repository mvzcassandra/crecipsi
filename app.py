# ══════════════════════════════════════════════════════════════
# CreciPSI v5.0 — Diseño final completo
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

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
#MainMenu,footer,.stDeployButton{display:none}

.header{
    background:linear-gradient(135deg,#0d3320 0%,#1a4731 60%,#2d6a4a 100%);
    padding:1.5rem 2rem;border-radius:14px;
    color:white;margin-bottom:1.5rem;
    display:flex;align-items:center;justify-content:space-between;
}
.header-left h1{font-size:1.8rem;font-weight:700;margin:0;letter-spacing:-0.5px}
.header-left p{font-size:0.85rem;opacity:0.75;margin:0.2rem 0 0}
.header-right{text-align:right}
.badge{
    display:inline-block;background:rgba(255,255,255,0.15);
    padding:0.25rem 0.85rem;border-radius:20px;
    font-size:0.72rem;letter-spacing:0.3px;
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
    font-size:0.75rem;font-weight:600;color:#1a4731;
    margin-bottom:0.75rem;
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
.ind-warn .ind-dot{background:#d97706}
.ind-warn{color:#7c4a00}
.ind-alert .ind-dot{background:#dc2626}
.ind-alert{color:#7f1d1d}
.ind-info .ind-dot{background:#2563eb}
.ind-info{color:#1e3a8a}
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
div[data-testid="stNumberInput"] input{
    border-radius:6px!important;font-size:0.9rem!important;
    text-align:center!important;
}
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
  <div class="header-left">
    <h1>🐴 CreciPSI</h1>
    <p>Monitor Inteligente de Crecimiento · Pura Sangre Inglés</p>
  </div>
  <div class="header-right">
    <span class="badge">FMVZ-UNAM</span><br>
    <span style="font-size:0.72rem;opacity:0.65;margin-top:4px;display:block">
      Diplomado IA en Salud Global · 2025–2026
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Métricas
st.markdown("""
<div class="metric-strip">
  <div class="metric-box"><div class="mv">217</div><div class="ml">Potros PSI</div></div>
  <div class="metric-box"><div class="mv">4,175</div><div class="ml">Mediciones</div></div>
  <div class="metric-box"><div class="mv">10 años</div><div class="ml">2015–2025</div></div>
  <div class="metric-box"><div class="mv">0.964</div><div class="ml">R² Modelo</div></div>
  <div class="metric-box"><div class="mv">15 kg</div><div class="ml">Error medio</div></div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Curvas de Referencia",
    "🔍 Evaluar un Potro",
    "🎯 Predictor",
    "ℹ️ Metodología"
])


# ══════════════════════════════════════════════════════════════
# HELPER: graficar curvas
# ══════════════════════════════════════════════════════════════

def fig_curvas(stats, color, titulo, ylabel, ylim,
               meses_anotados, fmt_anotacion, offset_anotacion,
               pesos_potro=None, nombre_potro=None,
               alzadas_potro=None, es_alzada=False):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor("#f8faf8")
    ax.set_facecolor("#f8faf8")
    edades = stats["edad_meses"]

    ax.fill_between(edades, stats.p10, stats.p90,
                    alpha=0.08, color=color)
    ax.fill_between(edades, stats.p25, stats.p75,
                    alpha=0.22, color=color, label="Rango normal (P25–P75)")
    ax.plot(edades, stats.p50, color=color,
            linewidth=2.2, label="Mediana (P50)", zorder=3)
    ax.plot(edades, stats.p10, color=color,
            linewidth=0.8, linestyle=":", alpha=0.45, label="P10 / P90")
    ax.plot(edades, stats.p90, color=color,
            linewidth=0.8, linestyle=":", alpha=0.45)

    for mes in meses_anotados:
        f = stats[stats.edad_meses == mes]
        if len(f) == 0: continue
        v = f["p50"].values[0]
        ax.annotate(
            fmt_anotacion.format(v),
            xy=(mes, v), xytext=(mes + 0.5, v + offset_anotacion),
            fontsize=8, color=color, fontweight="600",
            arrowprops=dict(arrowstyle="->", color=color, lw=1),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=color, alpha=0.85)
        )

    # Curva del potro evaluado (si existe)
    if pesos_potro and len(pesos_potro) >= 2:
        edades_p = sorted(pesos_potro.keys())
        vals_p   = [pesos_potro[e] for e in edades_p]
        ax.plot(edades_p, vals_p,
                color="#e67e22", linewidth=2.5,
                marker="o", markersize=6,
                label=nombre_potro or "Potro evaluado",
                zorder=5)

    ax.set_xlabel("Edad (meses)", fontsize=10, color="#4a4a4a")
    ax.set_ylabel(ylabel, fontsize=10, color="#4a4a4a")
    ax.set_title(titulo, fontsize=11, fontweight="700",
                 color="#1a2e1a", pad=10)
    ax.legend(fontsize=8.5, framealpha=0.9,
              edgecolor="#d4e8d8", loc="upper left")
    ax.grid(True, alpha=0.15, color="#94a3b8", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d4e8d8")
    ax.spines["bottom"].set_color("#d4e8d8")
    ax.set_xlim(-0.3 if 0 in edades.values else 0.5, 22.5)
    ax.set_ylim(*ylim)
    ax.tick_params(colors="#6b7280", labelsize=8.5)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 1 — CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-pill">📊 Curvas del rancho 2015–2025</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        sx1 = st.radio("Sexo:", ["Machos ♂", "Hembras ♀"],
                       horizontal=True, key="sx1")
    with c2:
        vr1 = st.radio("Variable:", ["Peso (kg)", "Alzada (m)"],
                       horizontal=True, key="vr1")

    sk1   = "M" if "Machos" in sx1 else "H"
    col1  = C[sk1]
    n1    = 111 if sk1 == "M" else 106

    if "Peso" in vr1:
        st_d = stats_ref[f"stats_{sk1}"]
        fig1 = fig_curvas(
            st_d, col1,
            f"Curvas de Peso — {'Machos' if sk1=='M' else 'Hembras'} PSI  (n={n1} animales)",
            "Peso (kg)", (20, 570),
            [0, 6, 12, 18], "{:.0f} kg", 20
        )
    else:
        st_d = stats_alz[f"stats_{sk1}"]
        fig1 = fig_curvas(
            st_d, col1,
            f"Curvas de Alzada — {'Machos' if sk1=='M' else 'Hembras'} PSI  (n={n1} animales)",
            "Alzada (m)", (0.85, 1.68),
            [6, 12, 18], "{:.2f} m", 0.012
        )

    st.pyplot(fig1)
    plt.close(fig1)

    with st.expander("Ver tabla de valores de referencia"):
        t1 = st_d[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
        t1.columns = ["Edad","P10","P25","P50","P75","P90","N"]
        dec = 1 if "Peso" in vr1 else 3
        st.dataframe(t1.round(dec),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.83rem;color:#2d5a3d;margin-top:0.5rem">
        <strong>Cómo leer:</strong> La zona sombreada oscura es el rango normal
        (P25–P75) donde se encuentra el 50% central de la población.
        Las líneas punteadas marcan P10 y P90.
        Un potro fuera de ese rango merece seguimiento clínico.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-pill">🔍 Evaluación individual</div>',
                unsafe_allow_html=True)

    # Datos básicos
    col_n, col_s = st.columns([2, 1])
    with col_n:
        nombre2 = st.text_input("Nombre / identificador",
                                placeholder="Ej. Hijo de Mila Race",
                                key="nombre2")
    with col_s:
        sx2 = st.radio("Sexo:", ["Macho ♂", "Hembra ♀"],
                       horizontal=True, key="sx2")
    sk2 = "M" if "Macho" in sx2 else "H"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#6b8f72;margin-bottom:0.75rem">
        Ingresa <strong>peso (kg)</strong> y/o <strong>alzada (m)</strong>
        en los meses disponibles. Mínimo 2 meses con peso.
        Deja en 0 los meses sin medición.
    </div>
    """, unsafe_allow_html=True)

    # Meses clave visibles por defecto
    MESES_CLAVE = [1, 3, 6, 9, 12, 18]
    MESES_EXTRA = [2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22]

    pesos2 = {}
    alzadas2 = {}

    # Peso al nacer aparte
    pn_col, _ = st.columns([1, 3])
    with pn_col:
        pnac2 = st.number_input("Peso al nacer (kg) — opcional",
                                min_value=0.0, max_value=80.0,
                                value=0.0, step=0.5, key="pnac2")
    if pnac2 > 0:
        pesos2[0] = pnac2

    # Grid de meses clave
    cols_k = st.columns(3)
    for i, mes in enumerate(MESES_CLAVE):
        with cols_k[i % 3]:
            with st.container():
                st.markdown(f"**Mes {mes}**")
                ca, cb = st.columns(2)
                with ca:
                    pv = st.number_input(
                        "Peso kg", min_value=0.0, max_value=700.0,
                        value=0.0, step=1.0, key=f"p2_{mes}",
                        label_visibility="visible"
                    )
                with cb:
                    av = st.number_input(
                        "Alzada m", min_value=0.0, max_value=2.0,
                        value=0.0, step=0.01, key=f"a2_{mes}",
                        label_visibility="visible"
                    )
                if pv > 0: pesos2[mes]   = pv
                if av > 0: alzadas2[mes] = av

    # Meses adicionales
    with st.expander("➕ Agregar más meses"):
        cols_e = st.columns(4)
        for i, mes in enumerate(MESES_EXTRA):
            with cols_e[i % 4]:
                st.markdown(f"**Mes {mes}**")
                ca2, cb2 = st.columns(2)
                with ca2:
                    pv2 = st.number_input(
                        "kg", min_value=0.0, max_value=700.0,
                        value=0.0, step=1.0, key=f"p2_{mes}",
                        label_visibility="visible"
                    )
                with cb2:
                    av2 = st.number_input(
                        "m", min_value=0.0, max_value=2.0,
                        value=0.0, step=0.01, key=f"a2_{mes}",
                        label_visibility="visible"
                    )
                if pv2 > 0: pesos2[mes]   = pv2
                if av2 > 0: alzadas2[mes] = av2

    st.markdown("<hr>", unsafe_allow_html=True)
    analizar2 = st.button("🔍 Ver evaluación completa",
                          type="primary", use_container_width=True,
                          key="btn2")

    if analizar2:
        meses_p = sorted([m for m in pesos2 if pesos2[m] > 0])
        if len(meses_p) < 2:
            st.warning("Ingresa al menos 2 mediciones de peso para evaluar.")
            st.stop()

        sp2 = stats_ref[f"stats_{sk2}"]
        sa2 = stats_alz[f"stats_{sk2}"]
        col2 = C[sk2]
        nombre_display = nombre2 or "Potro evaluado"

        # Evaluación mes a mes
        filas2 = []
        alertas2 = []
        alto_cnt = bajo_cnt = 0

        for mes in meses_p:
            if mes == 0: continue
            idx = mes - 1
            peso = pesos2[mes]
            ref  = sp2[sp2.edad_meses == mes]
            if ref.empty: continue

            p10=ref["p10"].values[0]; p25=ref["p25"].values[0]
            p50=ref["p50"].values[0]; p75=ref["p75"].values[0]
            p90=ref["p90"].values[0]
            diff = ((peso - p50) / p50) * 100

            if peso < p10:    zona="MUY BAJO"; alerta=True;  bajo_cnt += 1
            elif peso < p25:  zona="BAJO";     alerta=True;  bajo_cnt += 1
            elif peso <= p75: zona="NORMAL";   alerta=False
            elif peso <= p90: zona="ALTO";     alerta=False; alto_cnt += 1
            else:             zona="MUY ALTO"; alerta=True;  alto_cnt += 1

            if alerta: alertas2.append(mes)

            # Alzada si existe
            alz_zona = None
            if mes in alzadas2 and alzadas2[mes] > 0:
                ref_a = sa2[sa2.edad_meses == mes]
                if not ref_a.empty:
                    ap25=ref_a["p25"].values[0]
                    ap75=ref_a["p75"].values[0]
                    if alzadas2[mes] < ap25:    alz_zona = "Baja"
                    elif alzadas2[mes] <= ap75: alz_zona = "Normal"
                    else:                        alz_zona = "Alta"

            filas2.append({
                "mes": mes, "peso": peso,
                "p10":p10,"p25":p25,"p50":p50,"p75":p75,"p90":p90,
                "diff": diff, "zona": zona, "alerta": alerta,
                "alzada": alzadas2.get(mes, None), "alz_zona": alz_zona
            })

        # Clasificar patrón
        n_filas = len(filas2)
        pct_alto = alto_cnt / n_filas if n_filas > 0 else 0
        pct_bajo = bajo_cnt / n_filas if n_filas > 0 else 0
        vals_p = [f["peso"] for f in filas2]
        perdidas = sum(1 for i in range(1, len(vals_p))
                       if vals_p[i] < vals_p[i-1])
        caida = any((vals_p[i]-vals_p[i-1])/vals_p[i-1]*100 < -8
                    for i in range(1, len(vals_p)))

        if (perdidas >= 4) or caida:
            patron2 = "Patrón Irregular"; cls2 = "p-irregular"
            desc2   = "Pérdidas de peso detectadas entre mediciones consecutivas."
            indicaciones2 = [
                ("alert","Evaluación clínica urgente — pérdida de peso en potro en crecimiento."),
                ("alert","Descartar enfermedades gastrointestinales, parasitosis o estrés."),
                ("warn", "Revisar calidad y cantidad del alimento ofrecido."),
                ("warn", "Verificar disponibilidad de agua limpia y acceso al comedero."),
            ]
        elif pct_alto >= 0.60:
            patron2 = "Patrón Superior"; cls2 = "p-superior"
            desc2   = "Crecimiento consistentemente por encima del promedio del rancho."
            indicaciones2 = [
                ("info","Crecimiento excelente — por encima del P75 en la mayoría de los meses."),
                ("",    "Mantener el plan nutricional y de manejo actual."),
                ("warn","Vigilar condición corporal para evitar sobrepeso en meses tardíos."),
            ]
        elif pct_bajo >= 0.60:
            patron2 = "Patrón Inferior"; cls2 = "p-inferior"
            desc2   = "Crecimiento persistentemente por debajo del rango esperado."
            indicaciones2 = [
                ("warn","Revisar aporte energético — considerar incrementar concentrado."),
                ("warn","Evaluar estado de desparasitación — alta carga reduce absorción."),
                ("warn","Verificar disponibilidad de forraje y agua limpia."),
                ("",    "Repetir evaluación en 4 semanas tras ajuste nutricional."),
            ]
        else:
            patron2 = "Patrón Normal"; cls2 = "p-normal"
            desc2   = "Crecimiento dentro del rango esperado para el rancho."
            indicaciones2 = [
                ("",   "Mantener el programa de manejo y alimentación actual."),
                ("",   "Continuar con pesajes mensuales para seguimiento."),
            ]
            if alertas2:
                indicaciones2.append(
                    ("warn", f"Vigilar meses con alerta: {alertas2}.")
                )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Resultado principal
        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f"""
            <div class="patron-box {cls2}">{patron2}</div>
            <div style="font-size:0.83rem;color:#4a6a4a;margin-bottom:0.75rem">
                {desc2}
            </div>
            """, unsafe_allow_html=True)

            # Stats
            ganancia = round(vals_p[-1] - vals_p[0]) if len(vals_p) >= 2 else 0
            normal_pct = round(
                sum(1 for f in filas2 if f["zona"] == "NORMAL") / n_filas * 100
            )
            cls_n = "ok" if normal_pct >= 60 else ("warn" if normal_pct >= 40 else "bad")
            cls_a = "" if len(alertas2) == 0 else ("warn" if len(alertas2) <= 2 else "bad")

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card {cls_n}">
                <div class="sv">{normal_pct}%</div>
                <div class="sl">En rango normal</div>
              </div>
              <div class="stat-card">
                <div class="sv">+{ganancia} kg</div>
                <div class="sl">Ganancia total</div>
              </div>
              <div class="stat-card {cls_a}">
                <div class="sv">{len(alertas2)}</div>
                <div class="sl">Alertas</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Alertas
            if len(alertas2) == 0:
                st.markdown('<div class="ok-strip">✅ Sin alertas — crecimiento dentro del rango en todos los meses</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-strip">⚠️ Alertas en meses: {alertas2}</div>',
                            unsafe_allow_html=True)

            # Indicaciones clínicas
            ind_html = "".join([
                f'<div class="ind-item ind-{c if c else "ok"}">'
                f'<div class="ind-dot"></div><span>{t}</span></div>'
                for c, t in indicaciones2
            ])
            st.markdown(f"""
            <div class="ind-box">
              <div class="ind-titulo">Indicaciones clínicas</div>
              {ind_html}
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Gráfica de peso con curvas de referencia
            pesos_potro_dict = {f["mes"]: f["peso"] for f in filas2}
            fig_p = fig_curvas(
                sp2, col2,
                f"Peso — {nombre_display} vs. Curvas de referencia",
                "Peso (kg)", (50, max(max(pesos_potro_dict.values())+50, 520)),
                [6, 12, 18], "{:.0f} kg", 20,
                pesos_potro=pesos_potro_dict,
                nombre_potro=nombre_display
            )
            st.pyplot(fig_p)
            plt.close(fig_p)

            # Gráfica de alzada si hay datos
            alzadas_dict = {mes: alzadas2[mes] for mes in alzadas2
                           if alzadas2[mes] > 0 and mes > 0}
            if len(alzadas_dict) >= 2:
                st.markdown("**Alzada**", unsafe_allow_html=False)
                fig_a = fig_curvas(
                    sa2, col2,
                    f"Alzada — {nombre_display} vs. Curvas de referencia",
                    "Alzada (m)", (0.85, 1.70),
                    [6, 12, 18], "{:.2f} m", 0.012,
                    pesos_potro=alzadas_dict,
                    nombre_potro=nombre_display,
                    es_alzada=True
                )
                st.pyplot(fig_a)
                plt.close(fig_a)

        # Tabla detallada
        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("Ver evaluación detallada mes a mes"):
            df2 = pd.DataFrame([{
                "Mes": f["mes"],
                "Peso real (kg)": f["peso"],
                "P25": round(f["p25"],1),
                "P50": round(f["p50"],1),
                "P75": round(f["p75"],1),
                "vs P50": f'{f["diff"]:+.1f}%',
                "Estado peso": f["zona"],
                "Alzada (m)": f["alzada"] if f["alzada"] else "—",
                "Estado alzada": f["alz_zona"] if f["alz_zona"] else "—",
            } for f in filas2])

            def color_estado(v):
                if "BAJO" in str(v):   return "background:#fff3e0;color:#92400e"
                elif "ALTO" in str(v): return "background:#dbeafe;color:#1e40af"
                elif "NORMAL" in str(v):return "background:#e8f4ee;color:#1a4731"
                return ""

            st.dataframe(
                df2.style.applymap(color_estado, subset=["Estado peso"]),
                use_container_width=True, hide_index=True
            )


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-pill">🎯 Predictor de peso y alzada</div>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([1, 1, 1])
    with pc1:
        sx3 = st.radio("Sexo:", ["Macho ♂","Hembra ♀"],
                       horizontal=True, key="sx3")
    with pc2:
        edad3 = st.slider("Edad (meses):", 1, 22, 6, key="edad3")
    with pc3:
        alz3 = st.number_input(
            "Alzada actual (m) — opcional",
            min_value=0.0, max_value=2.0,
            value=0.0, step=0.01, key="alz3",
            help="Si la ingresas el modelo es más preciso (R²=0.9641)"
        )

    sk3    = "M" if "Macho" in sx3 else "H"
    sbin3  = 1 if sk3 == "M" else 0
    col3   = C[sk3]

    sp3    = stats_ref[f"stats_{sk3}"]
    sa3    = stats_alz[f"stats_{sk3}"]
    ref_a3 = sa3[sa3.edad_meses == edad3]
    alz_m3 = ref_a3["p50"].values[0] if len(ref_a3) > 0 else 1.35
    alz_u3 = alz3 if alz3 > 0 else alz_m3

    peso3  = mod_peso.predict([[sbin3, edad3, alz_u3]])[0]
    alzp3  = mod_alz.predict([[sbin3, edad3, peso3]])[0]
    ref_p3 = sp3[sp3.edad_meses == edad3]

    st.markdown("<hr>", unsafe_allow_html=True)

    if alz3 > 0:
        st.markdown('<div class="ok-strip">Usando modelo mejorado con alzada — R²=0.9641</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#f0f9ff;border:0.5px solid #bae6fd;border-radius:8px;'
            'padding:0.6rem 1rem;font-size:0.82rem;color:#0c4a6e;margin-bottom:0.5rem">'
            'Ingresa la alzada para mayor precisión (R²=0.9641 vs 0.9458)</div>',
            unsafe_allow_html=True
        )

    # Resultados
    if len(ref_p3) > 0:
        p25r = ref_p3["p25"].values[0]
        p50r = ref_p3["p50"].values[0]
        p75r = ref_p3["p75"].values[0]

        if peso3 < p25r:    pos="Inferior al rango normal"; pc="#92400e"; pb="#fff3e0"
        elif peso3 <= p75r: pos="Dentro del rango normal";  pc="#1a4731"; pb="#e8f4ee"
        else:               pos="Superior al rango normal"; pc="#1e40af"; pb="#dbeafe"

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(f"""
            <div style="background:{pb};border:1.5px solid {pc};border-radius:10px;
                        padding:1rem;text-align:center">
              <div style="font-size:2rem;font-weight:700;color:{pc}">{peso3:.0f} kg</div>
              <div style="font-size:0.78rem;color:{pc};opacity:0.85;margin-top:3px">
                Peso predicho · mes {edad3}
              </div>
              <div style="font-size:0.72rem;color:{pc};opacity:0.7;margin-top:4px;
                          background:rgba(255,255,255,0.5);border-radius:6px;padding:2px 6px">
                {pos}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.metric("P25 del rancho", f"{p25r:.0f} kg")
        with rc3:
            st.metric("P50 del rancho", f"{p50r:.0f} kg")
        with rc4:
            st.metric("P75 del rancho", f"{p75r:.0f} kg")

        st.markdown("")

        if len(ref_a3) > 0:
            a25=ref_a3["p25"].values[0]; a50=ref_a3["p50"].values[0]
            a75=ref_a3["p75"].values[0]
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

    # Gráficas de curvas completas con punto marcado
    edades_g = list(range(1, 23))
    saM3=stats_alz["stats_M"]; saH3=stats_alz["stats_H"]
    spM3=stats_ref["stats_M"]; spH3=stats_ref["stats_H"]

    aM3 = [saM3[saM3.edad_meses==e]["p50"].values[0]
           if len(saM3[saM3.edad_meses==e])>0 else 1.35 for e in edades_g]
    aH3 = [saH3[saH3.edad_meses==e]["p50"].values[0]
           if len(saH3[saH3.edad_meses==e])>0 else 1.33 for e in edades_g]

    pM3=[mod_peso.predict([[1,e,a]])[0] for e,a in zip(edades_g,aM3)]
    pH3=[mod_peso.predict([[0,e,a]])[0] for e,a in zip(edades_g,aH3)]

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.5))
    fig3.patch.set_facecolor("#f8faf8")

    for ax3, (spX, preds, colX, lab) in zip(axes3, [
        (spM3, pM3, C["M"], "Machos"),
        (spH3, pH3, C["H"], "Hembras")
    ]):
        ax3.set_facecolor("#f8faf8")
        ax3.fill_between(spX.edad_meses, spX.p25, spX.p75,
                         alpha=0.20, color=colX, label="Rango normal")
        ax3.fill_between(spX.edad_meses, spX.p10, spX.p90,
                         alpha=0.07, color=colX)
        ax3.plot(edades_g, preds, color=colX,
                 linewidth=2.2, label=f"Predicción {lab}")
        ax3.plot(spX.edad_meses, spX.p50,
                 color=colX, linewidth=1.2,
                 linestyle="--", alpha=0.5, label="Mediana real")
        if sk3 == ("M" if colX==C["M"] else "H"):
            ax3.axvline(x=edad3, color="#94a3b8",
                        linestyle="--", linewidth=1.2, alpha=0.8)
            ax3.scatter([edad3], [peso3], color="#e67e22",
                        s=160, zorder=6, edgecolors="white",
                        linewidths=2, label=f"{peso3:.0f} kg")
        ax3.set_xlabel("Edad (meses)", fontsize=10)
        ax3.set_ylabel("Peso (kg)", fontsize=10)
        ax3.set_title(f"{lab}", fontsize=11, fontweight="700")
        ax3.legend(fontsize=8, framealpha=0.9)
        ax3.grid(True, alpha=0.15, linestyle="--")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.tick_params(labelsize=8.5)

    fig3.suptitle("Curvas de predicción vs. rangos normales del rancho",
                  fontsize=11, fontweight="700", y=1.02)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    with st.expander("Ver tabla completa de predicciones"):
        st.dataframe(pd.DataFrame({
            "Mes":               edades_g,
            "Peso Machos (kg)":  [round(p) for p in pM3],
            "Peso Hembras (kg)": [round(p) for p in pH3],
            "Alzada Machos (m)": [round(a,3) for a in aM3],
            "Alzada Hembras (m)":[round(a,3) for a in aH3],
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-pill">ℹ️ Metodología y referencias</div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### Base de datos")
        st.markdown("""
        Registros zootécnicos reales de un rancho PSI mexicano
        (2015–2025). Dataset de **217 animales** con **4,175 mediciones**
        de peso y **3,981 de alzada**. Completitud del 100% en alzada.
        El 89.4% de los animales tienen peso al nacer registrado.
        """)
        st.markdown("#### Estadística aplicada")
        st.markdown("""
        - Percentiles P10–P90 por edad (0–22 meses) y sexo
        - Regresión polinomial grado 3 con variables sexo, edad y alzada
        - Validación train/test 80%/20% con semilla fija (random_state=42)
        - Clasificador basado en criterios clínicos equinos
        - Correlación peso-alzada: r=0.9666 (Pearson)
        """)

    with m2:
        st.markdown("#### Métricas de validación")
        st.dataframe(pd.DataFrame({
            "Modelo":    ["Peso (con alzada)","Peso (sin alzada)","Alzada"],
            "R²":        ["0.9641","0.9458","0.9552"],
            "MAE":       ["15.1 kg","19.6 kg","2.0 cm"],
            "N datos":   ["3,181","3,181","3,181"],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### Limitaciones")
        st.markdown("""
        - Curvas específicas para esta población y rancho
        - No generalizable sin validación externa
        - Sin variables de alimentación o sanidad
        - 10.6% sin peso al nacer (principalmente cohorte 2015)
        """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Referencias (Vancouver)")
    st.markdown("""
    1. Hintz HF, Hintz RL, Van Vleck LD. Growth rate of thoroughbreds. *J Anim Sci.* 1979;48(3):480-487.
    2. National Research Council. *Nutrient Requirements of Horses.* 6th ed. Washington DC: NAP; 2007.
    3. WHO Multicentre Growth Reference Study Group. *WHO child growth standards.* Geneva: WHO; 2006.
    4. James G, Witten D, Hastie T, Tibshirani R. *An Introduction to Statistical Learning.* 2nd ed. Springer; 2021.
    5. Dohoo I, Martin W, Stryhn H. *Veterinary Epidemiologic Research.* 2nd ed. VER Inc; 2009.
    6. Staniar WB et al. Growth trajectory of thoroughbreds in Kentucky. *J Anim Sci.* 2004;82(8):2352-2362.
    """)

    st.markdown("""
    <div style="background:#f0faf4;border:0.5px solid #52b788;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.83rem;color:#2d5a3d;margin-top:0.5rem">
        <strong>Citación sugerida:</strong> [Autor]. CreciPSI: Sistema de monitoreo
        inteligente de crecimiento en potros Pura Sangre Inglés mediante inteligencia
        artificial. FMVZ-UNAM. Diplomado en IA en Salud Global. 2026.
    </div>
    """, unsafe_allow_html=True)
