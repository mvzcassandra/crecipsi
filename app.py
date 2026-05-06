# ══════════════════════════════════════════════════════════════
# CreciPSI v3.0 — UI/UX profesional
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

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y ESTILOS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CreciPSI",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para UI/UX profesional
st.markdown("""
<style>
    /* Tipografia base */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0.4rem 0 0 0;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(10px);
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f3460;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Patron de crecimiento */
    .patron-normal    { background:#dcfce7; color:#166534; border:1.5px solid #86efac; }
    .patron-superior  { background:#dbeafe; color:#1e3a8a; border:1.5px solid #93c5fd; }
    .patron-inferior  { background:#ffedd5; color:#9a3412; border:1.5px solid #fdba74; }
    .patron-irregular { background:#fee2e2; color:#991b1b; border:1.5px solid #fca5a5; }

    .patron-box {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin: 0.8rem 0;
    }

    /* Secciones */
    .section-header {
        border-left: 4px solid #0f3460;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }
    .section-header h2 {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    .section-header p {
        font-size: 0.88rem;
        color: #64748b;
        margin: 0.2rem 0 0 0;
    }

    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        color: #0c4a6e;
        margin: 0.8rem 0;
    }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        color: #92400e;
        margin: 0.8rem 0;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        color: #14532d;
        margin: 0.8rem 0;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        font-size: 0.84rem;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Inputs mas limpios */
    .stNumberInput > div > div > input {
        border-radius: 6px !important;
    }

    /* Divisor */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CARGAR MODELOS
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando modelos de IA...")
def cargar_modelos():
    with open("stats_ref_final.pkl",    "rb") as f: stats_ref    = pickle.load(f)
    with open("stats_alzada_final.pkl", "rb") as f: stats_alzada = pickle.load(f)
    with open("modelo_peso_v2.pkl",     "rb") as f: modelo_peso  = pickle.load(f)
    with open("modelo_alzada.pkl",      "rb") as f: modelo_alz   = pickle.load(f)
    return stats_ref, stats_alzada, modelo_peso, modelo_alz

try:
    stats_ref, stats_alzada, modelo_peso, modelo_alzada = cargar_modelos()
    ok = True
except Exception as e:
    ok = False
    err = str(e)


# ══════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🐴 CreciPSI</h1>
    <p>Monitor Inteligente de Crecimiento en Potros Pura Sangre Inglés</p>
    <span class="badge">FMVZ-UNAM &nbsp;·&nbsp; Diplomado IA en Salud Global &nbsp;·&nbsp; 2025–2026</span>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"Error al cargar los modelos: {err}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# MÉTRICAS GLOBALES
# ══════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)
metricas = [
    ("217", "Potros PSI"),
    ("4,175", "Mediciones"),
    ("10 años", "2015–2025"),
    ("0.9641", "R² Modelo"),
    ("15.1 kg", "Error promedio"),
]
cols = [c1, c2, c3, c4, c5]
for col, (val, lab) in zip(cols, metricas):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{lab}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Navegación")
    seccion = st.radio(
        "",
        ["📊 Curvas de Referencia",
         "🔍 Evaluar un Potro",
         "🎯 Predictor Inteligente",
         "ℹ️ Acerca del Sistema"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <strong>Modelos activos</strong><br>
        Peso (con alzada): R²=0.9641<br>
        Alzada: R²=0.9552<br>
        Rango válido: 0–22 meses
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <strong>Población de referencia</strong><br>
        217 potros PSI<br>
        Rancho mexicano 2015–2025<br>
        111 machos · 106 hembras
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <strong>Desarrollado por</strong><br>
        MVZ Cassandra<br>
        FMVZ-UNAM · 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PALETA DE COLORES CONSISTENTE
# ══════════════════════════════════════════════════════════════

COLORES = {
    "M":         "#0f3460",   # azul oscuro para machos
    "H":         "#831843",   # rosa oscuro para hembras
    "normal":    "#16a34a",
    "superior":  "#1d4ed8",
    "inferior":  "#ea580c",
    "irregular": "#dc2626",
    "alerta":    "#ef4444",
    "neutro":    "#64748b",
}


# ══════════════════════════════════════════════════════════════
# FUNCIÓN: Graficar curvas de referencia
# ══════════════════════════════════════════════════════════════

def graficar_curvas(stats, sexo_key, variable, unidad, titulo,
                    ylim, anotaciones, color):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    edades = stats['edad_meses']

    ax.fill_between(edades, stats.p10, stats.p90,
                    alpha=0.08, color=color)
    ax.fill_between(edades, stats.p25, stats.p75,
                    alpha=0.20, color=color,
                    label='Rango normal (P25–P75)')
    ax.plot(edades, stats.p50, color=color,
            linewidth=2.5, label='Mediana (P50)', zorder=3)
    ax.plot(edades, stats.p10, color=color,
            linewidth=0.8, linestyle=':', alpha=0.5, label='P10 / P90')
    ax.plot(edades, stats.p90, color=color,
            linewidth=0.8, linestyle=':', alpha=0.5)

    for mes_ref, offset in anotaciones:
        fila = stats[stats.edad_meses == mes_ref]
        if len(fila) > 0:
            p50_val = fila['p50'].values[0]
            fmt = f'{p50_val:.0f} {unidad}' if unidad == 'kg' \
                  else f'{p50_val:.2f} {unidad}'
            ax.annotate(
                fmt,
                xy=(mes_ref, p50_val),
                xytext=(mes_ref + 0.5, p50_val + offset),
                fontsize=8.5, color=color, fontweight='600',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor=color,
                          alpha=0.8)
            )

    ax.set_xlabel('Edad (meses)', fontsize=11, color='#374151')
    ax.set_ylabel(f'{variable} ({unidad})', fontsize=11, color='#374151')
    ax.set_title(titulo, fontsize=12, fontweight='700',
                 color='#1e293b', pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='#e2e8f0')
    ax.grid(True, alpha=0.2, color='#94a3b8', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.set_xlim(-0.3 if 0 in edades.values else 1, 22.3)
    ax.set_ylim(*ylim)
    ax.tick_params(colors='#6b7280', labelsize=9)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SECCIÓN 1: CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════

if seccion == "📊 Curvas de Referencia":

    st.markdown("""
    <div class="section-header">
        <h2>📊 Curvas de Referencia</h2>
        <p>Percentiles de peso y alzada calculados con datos reales del rancho</p>
    </div>
    """, unsafe_allow_html=True)

    # Selector de sexo y variable
    col_sx, col_var, _ = st.columns([1, 1, 2])
    with col_sx:
        sexo_sel = st.radio("Sexo:", ["Machos ♂", "Hembras ♀"],
                            horizontal=True, key="sexo_curvas")
    with col_var:
        var_sel = st.radio("Variable:", ["Peso (kg)", "Alzada (m)"],
                           horizontal=True, key="var_curvas")

    sexo_key = "M" if "Machos" in sexo_sel else "H"
    color    = COLORES[sexo_key]

    if "Peso" in var_sel:
        stats = stats_ref[f"stats_{sexo_key}"]
        fig = graficar_curvas(
            stats, sexo_key, "Peso", "kg",
            f"Curvas de Peso — {'Machos' if sexo_key=='M' else 'Hembras'} PSI "
            f"(n={'111' if sexo_key=='M' else '106'} animales)",
            (20, 570),
            [(0, 4), (6, 22), (12, 22), (18, 22)],
            color
        )
    else:
        stats = stats_alzada[f"stats_{sexo_key}"]
        fig = graficar_curvas(
            stats, sexo_key, "Alzada", "m",
            f"Curvas de Alzada — {'Machos' if sexo_key=='M' else 'Hembras'} PSI "
            f"(n={'111' if sexo_key=='M' else '106'} animales)",
            (0.85, 1.68),
            [(6, 0.012), (12, 0.012), (18, 0.012)],
            color
        )

    st.pyplot(fig)
    plt.close(fig)

    # Tabla de referencia
    with st.expander("Ver tabla de valores de referencia", expanded=False):
        tabla = stats[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
        tabla.columns = ["Edad (meses)","P10","P25",
                         "P50 (mediana)","P75","P90","N animales"]
        decimales = 1 if "Peso" in var_sel else 3
        st.dataframe(tabla.round(decimales),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
        <strong>Cómo leer estas curvas:</strong> El área oscura representa el 
        rango normal (P25–P75), donde se encuentra el 50% central de la población. 
        Un potro en esa zona crece como la mayoría. Las líneas punteadas marcan 
        el P10 y P90 — solo el 10% más ligero/bajo o más pesado/alto queda fuera 
        de ese rango.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 2: EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════

elif seccion == "🔍 Evaluar un Potro":

    st.markdown("""
    <div class="section-header">
        <h2>🔍 Evaluar un Potro</h2>
        <p>Compara el crecimiento individual contra las curvas de referencia del rancho</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Datos básicos ──
    col_id, col_sx = st.columns([2, 1])
    with col_id:
        nombre = st.text_input("Nombre o identificador del potro",
                               placeholder="Ej. Hijo de Mila Race",
                               label_visibility="visible")
    with col_sx:
        sexo_sel = st.radio("Sexo:", ["Macho ♂", "Hembra ♀"],
                            horizontal=True, key="sexo_eval")
    sexo_key = "M" if "Macho" in sexo_sel else "H"

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Entrada de mediciones ──
    st.markdown("**Ingresa las mediciones disponibles** — deja en 0 los meses sin dato")

    # Peso al nacer aparte
    col_nac1, col_nac2, _ = st.columns([1, 1, 2])
    with col_nac1:
        peso_nac = st.number_input("Peso al nacer (kg)",
                                   min_value=0.0, max_value=80.0,
                                   value=0.0, step=0.5, key="p_0")

    pesos_input   = {}
    alzadas_input = {}
    if peso_nac > 0:
        pesos_input[0] = peso_nac

    # Mediciones mensuales en grid de 4 columnas
    st.markdown("**Mediciones mensuales:**")

    MESES_POR_FILA = 4
    for fila_inicio in range(1, 23, MESES_POR_FILA):
        cols = st.columns(MESES_POR_FILA)
        for i, col in enumerate(cols):
            mes = fila_inicio + i
            if mes > 22:
                break
            with col:
                with st.expander(f"Mes {mes}", expanded=True):
                    pv = st.number_input(
                        "Peso (kg)", min_value=0.0, max_value=700.0,
                        value=0.0, step=1.0, key=f"p_{mes}",
                        label_visibility="visible"
                    )
                    av = st.number_input(
                        "Alzada (m)", min_value=0.0, max_value=2.0,
                        value=0.0, step=0.01, key=f"a_{mes}",
                        label_visibility="visible"
                    )
                    if pv > 0: pesos_input[mes]   = pv
                    if av > 0: alzadas_input[mes] = av

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Calcular y mostrar resultados ──
    n_pesos   = len(pesos_input)
    n_alzadas = len(alzadas_input)

    if n_pesos < 2 and n_alzadas < 2:
        st.markdown("""
        <div class="info-box">
            ℹ️ Ingresa al menos <strong>2 mediciones</strong> de peso o alzada 
            para ver la evaluación.
        </div>
        """, unsafe_allow_html=True)
    else:
        stats_p = stats_ref[f"stats_{sexo_key}"]
        stats_a = stats_alzada[f"stats_{sexo_key}"]

        # ── Evaluar peso ──
        def evaluar_variable(mediciones, stats, campo):
            filas = []
            for edad in sorted(mediciones.keys()):
                valor = mediciones[edad]
                ref   = stats[stats.edad_meses == int(edad)]
                if ref.empty: continue
                p10 = ref["p10"].values[0]
                p25 = ref["p25"].values[0]
                p50 = ref["p50"].values[0]
                p75 = ref["p75"].values[0]
                p90 = ref["p90"].values[0]
                diff = ((valor - p50) / p50) * 100

                if valor < p10:
                    zona="MUY BAJO"; alerta=True
                elif valor < p25:
                    zona="BAJO"; alerta=True
                elif valor <= p75:
                    zona="NORMAL"; alerta=False
                elif valor <= p90:
                    zona="ALTO"; alerta=False
                else:
                    zona="MUY ALTO"; alerta=True

                filas.append({
                    "edad_meses":edad, campo:round(valor,2),
                    "P10":round(p10,2),"P25":round(p25,2),
                    "P50":round(p50,2),"P75":round(p75,2),"P90":round(p90,2),
                    "diff_pct":diff, "zona":zona, "alerta":alerta
                })
            return pd.DataFrame(filas)

        df_p = evaluar_variable(pesos_input,   stats_p, "peso_kg")
        df_a = evaluar_variable(alzadas_input, stats_a, "alzada_m")

        # ── Clasificar patrón ──
        def clasificar_patron(df, campo_valor):
            if len(df) < 2:
                return "Sin datos", "neutro"
            prop_alto = df["zona"].str.contains("ALTO").mean()
            prop_bajo = df["zona"].str.contains("BAJO").mean()
            vals = list(df[campo_valor])
            perdidas = sum(1 for i in range(1,len(vals))
                          if vals[i] < vals[i-1])
            caida = any((vals[i]-vals[i-1])/vals[i-1]*100 < -8
                        for i in range(1,len(vals)))
            if (perdidas >= 4) or caida:
                return "Irregular", "irregular"
            elif prop_alto >= 0.60:
                return "Superior", "superior"
            elif prop_bajo >= 0.60:
                return "Inferior", "inferior"
            return "Normal", "normal"

        patron_p, clase_p = clasificar_patron(df_p, "peso_kg")
        patron_a, clase_a = clasificar_patron(df_a, "alzada_m")

        # ── Mostrar resultado ──
        nombre_display = nombre if nombre else "Potro evaluado"

        res_cols = st.columns(2)
        with res_cols[0]:
            if len(df_p) >= 2:
                st.markdown(f"""
                <div class="patron-box patron-{clase_p}">
                    Peso — Patrón {patron_p}
                </div>
                """, unsafe_allow_html=True)
                n_alert_p = df_p["alerta"].sum()
                if n_alert_p == 0:
                    st.markdown('<div class="success-box">✅ Sin alertas de peso</div>',
                                unsafe_allow_html=True)
                else:
                    meses_a = df_p[df_p["alerta"]==True]["edad_meses"].tolist()
                    st.markdown(f'<div class="warning-box">⚠️ Alertas en meses: {meses_a}</div>',
                                unsafe_allow_html=True)

        with res_cols[1]:
            if len(df_a) >= 2:
                st.markdown(f"""
                <div class="patron-box patron-{clase_a}">
                    Alzada — Patrón {patron_a}
                </div>
                """, unsafe_allow_html=True)
                n_alert_a = df_a["alerta"].sum()
                if n_alert_a == 0:
                    st.markdown('<div class="success-box">✅ Sin alertas de alzada</div>',
                                unsafe_allow_html=True)
                else:
                    meses_b = df_a[df_a["alerta"]==True]["edad_meses"].tolist()
                    st.markdown(f'<div class="warning-box">⚠️ Alertas en meses: {meses_b}</div>',
                                unsafe_allow_html=True)

        # ── Gráficas ──
        color_ref = COLORES[sexo_key]
        color_p   = COLORES[clase_p] if clase_p != "neutro" else color_ref
        color_a   = COLORES[clase_a] if clase_a != "neutro" else color_ref

        n_graficas = (1 if len(df_p)>=1 else 0) + (1 if len(df_a)>=1 else 0)
        if n_graficas > 0:
            fig_cols = st.columns(n_graficas)
            col_idx  = 0

            for df_var, campo, stats_var, color_var, ylabel, fmt in [
                (df_p, "peso_kg",  stats_p, color_p, "Peso (kg)",     "{:.0f}"),
                (df_a, "alzada_m", stats_a, color_a, "Alzada (m)", "{:.3f}"),
            ]:
                if len(df_var) < 1:
                    continue
                with fig_cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    fig.patch.set_facecolor('#fafafa')
                    ax.set_facecolor('#fafafa')

                    ax.fill_between(stats_var.edad_meses,
                                    stats_var.p25, stats_var.p75,
                                    alpha=0.18, color=color_ref,
                                    label="P25–P75 referencia")
                    ax.fill_between(stats_var.edad_meses,
                                    stats_var.p10, stats_var.p90,
                                    alpha=0.07, color=color_ref)
                    ax.plot(stats_var.edad_meses, stats_var.p50,
                            color=color_ref, linewidth=1.8,
                            linestyle="--", alpha=0.6, label="Mediana")
                    ax.plot(df_var["edad_meses"], df_var[campo],
                            color=color_var, linewidth=2.5,
                            marker="o", markersize=6,
                            label=nombre_display, zorder=5)

                    alertas = df_var[df_var["alerta"]==True]
                    if len(alertas) > 0:
                        ax.scatter(alertas["edad_meses"], alertas[campo],
                                   color=COLORES["alerta"], s=100,
                                   marker="x", linewidths=2.5,
                                   zorder=6, label="Alerta")

                    ax.set_xlabel("Edad (meses)", fontsize=10)
                    ax.set_ylabel(ylabel, fontsize=10)
                    ax.set_title(f"{ylabel} — {nombre_display}",
                                 fontsize=11, fontweight='700')
                    ax.legend(fontsize=8, framealpha=0.9)
                    ax.grid(True, alpha=0.18, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                col_idx += 1

        # ── Tablas detalladas ──
        def tabla_detallada(df, campo, unidad):
            if len(df) == 0:
                return
            df_show = df[["edad_meses", campo,
                          "P25","P50","P75","zona"]].copy()
            df_show["diff"] = df["diff_pct"].apply(lambda x: f"{x:+.1f}%")
            df_show.columns = ["Mes", f"Real ({unidad})",
                                "P25","P50","P75","Estado","vs P50"]

            def color_estado(v):
                if "BAJO" in str(v):   return "background-color:#ffedd5;color:#9a3412"
                elif "ALTO" in str(v): return "background-color:#dbeafe;color:#1e3a8a"
                elif "NORMAL" in str(v):return "background-color:#dcfce7;color:#166534"
                return ""

            st.dataframe(
                df_show.style.applymap(color_estado, subset=["Estado"]),
                use_container_width=True, hide_index=True
            )

        if len(df_p) >= 1:
            with st.expander("Ver evaluación de peso mes a mes", expanded=False):
                tabla_detallada(df_p, "peso_kg", "kg")

        if len(df_a) >= 1:
            with st.expander("Ver evaluación de alzada mes a mes", expanded=False):
                tabla_detallada(df_a, "alzada_m", "m")


# ══════════════════════════════════════════════════════════════
# SECCIÓN 3: PREDICTOR INTELIGENTE
# ══════════════════════════════════════════════════════════════

elif seccion == "🎯 Predictor Inteligente":

    st.markdown("""
    <div class="section-header">
        <h2>🎯 Predictor Inteligente</h2>
        <p>Estima el peso y alzada esperados según edad, sexo y datos disponibles</p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_res = st.columns([1, 1])

    with col_inp:
        st.markdown("**Parámetros de entrada**")

        sexo_pred = st.radio("Sexo:", ["Macho ♂", "Hembra ♀"],
                             horizontal=True, key="sexo_pred")
        sexo_bin  = 1 if "Macho" in sexo_pred else 0
        sexo_key  = "M" if sexo_bin == 1 else "H"

        edad_pred = st.slider("Edad (meses):", 1, 22, 6,
                              help="Mueve el slider para cambiar la edad")

        alzada_inp = st.number_input(
            "Alzada actual (m) — opcional",
            min_value=0.0, max_value=2.0, value=0.0, step=0.01,
            help="Si la ingresas, el modelo es más preciso (R²=0.9641 vs 0.9458)"
        )
        usar_alzada = alzada_inp > 0

        if usar_alzada:
            st.markdown('<div class="success-box">Usando modelo mejorado con alzada — R²=0.9641</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Usando modelo base — R²=0.9458. Ingresa la alzada para mayor precisión.</div>',
                        unsafe_allow_html=True)

    with col_res:
        st.markdown("**Resultado**")

        stats_p_pred = stats_ref[f"stats_{sexo_key}"]
        stats_a_pred = stats_alzada[f"stats_{sexo_key}"]
        ref_alz      = stats_a_pred[stats_a_pred.edad_meses == edad_pred]
        alz_mediana  = ref_alz["p50"].values[0] if len(ref_alz) > 0 else 1.35

        alz_para_modelo = alzada_inp if usar_alzada else alz_mediana
        peso_pred = modelo_peso.predict(
            [[sexo_bin, edad_pred, alz_para_modelo]])[0]
        alz_pred  = modelo_alzada.predict(
            [[sexo_bin, edad_pred, peso_pred]])[0]

        ref_peso = stats_p_pred[stats_p_pred.edad_meses == edad_pred]

        if len(ref_peso) > 0:
            p25 = ref_peso["p25"].values[0]
            p50 = ref_peso["p50"].values[0]
            p75 = ref_peso["p75"].values[0]

            # Indicador visual de posición
            if peso_pred < p25:
                pos_txt = "Por debajo del rango normal"
                pos_col = COLORES["inferior"]
            elif peso_pred <= p75:
                pos_txt = "Dentro del rango normal"
                pos_col = COLORES["normal"]
            else:
                pos_txt = "Por encima del rango normal"
                pos_col = COLORES["superior"]

            st.markdown(f"""
            <div style="background:{pos_col};color:white;
                        border-radius:10px;padding:1rem 1.5rem;
                        text-align:center;margin-bottom:1rem">
                <div style="font-size:2.2rem;font-weight:800">
                    {peso_pred:.0f} kg
                </div>
                <div style="font-size:0.85rem;opacity:0.9">
                    Peso predicho a los {edad_pred} meses
                </div>
                <div style="font-size:0.8rem;margin-top:0.3rem;
                            background:rgba(255,255,255,0.2);
                            border-radius:6px;padding:0.2rem 0.5rem">
                    {pos_txt}
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("P25 rancho", f"{p25:.0f} kg")
            with m2:
                st.metric("P50 rancho", f"{p50:.0f} kg")
            with m3:
                st.metric("P75 rancho", f"{p75:.0f} kg")

        # Alzada predicha
        ref_alz_res = stats_a_pred[stats_a_pred.edad_meses == edad_pred]
        if len(ref_alz_res) > 0:
            a50 = ref_alz_res["p50"].values[0]
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.8rem">
                <div class="value">{alz_pred:.3f} m</div>
                <div class="label">Alzada predicha · Mediana rancho: {a50:.3f} m</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gráfica de curvas completas ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Curvas de predicción completas**")

    edades   = list(range(1, 23))
    stats_aM = stats_alzada["stats_M"]
    stats_aH = stats_alzada["stats_H"]
    stats_pM = stats_ref["stats_M"]
    stats_pH = stats_ref["stats_H"]

    alz_M = [stats_aM[stats_aM.edad_meses==e]["p50"].values[0]
              if len(stats_aM[stats_aM.edad_meses==e])>0 else 1.35 for e in edades]
    alz_H = [stats_aH[stats_aH.edad_meses==e]["p50"].values[0]
              if len(stats_aH[stats_aH.edad_meses==e])>0 else 1.33 for e in edades]

    pM = [modelo_peso.predict([[1, e, a]])[0] for e, a in zip(edades, alz_M)]
    pH = [modelo_peso.predict([[0, e, a]])[0] for e, a in zip(edades, alz_H)]

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    fig4.patch.set_facecolor('#fafafa')
    ax4.set_facecolor('#fafafa')

    ax4.fill_between(stats_pM.edad_meses, stats_pM.p25, stats_pM.p75,
                     alpha=0.12, color=COLORES["M"])
    ax4.fill_between(stats_pH.edad_meses, stats_pH.p25, stats_pH.p75,
                     alpha=0.12, color=COLORES["H"])
    ax4.plot(edades, pM, color=COLORES["M"], linewidth=2.5, label="Machos")
    ax4.plot(edades, pH, color=COLORES["H"], linewidth=2.5, label="Hembras")
    ax4.axvline(x=edad_pred, color="#94a3b8", linestyle="--",
                linewidth=1.5, alpha=0.8)
    ax4.scatter([edad_pred], [peso_pred], color="#f59e0b",
                s=180, zorder=6, edgecolors="white", linewidths=2,
                label=f"Predicción: {peso_pred:.0f} kg")

    ax4.set_xlabel("Edad (meses)", fontsize=11)
    ax4.set_ylabel("Peso (kg)", fontsize=11)
    ax4.set_title("Curvas de predicción vs. rangos normales del rancho",
                  fontsize=12, fontweight='700')
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.18, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    st.pyplot(fig4)
    plt.close(fig4)

    # Tabla completa
    with st.expander("Ver tabla completa de predicciones", expanded=False):
        df_tabla = pd.DataFrame({
            "Edad (meses)":       edades,
            "Peso Machos (kg)":   [round(p) for p in pM],
            "Peso Hembras (kg)":  [round(p) for p in pH],
            "Alzada Machos (m)":  [round(a,3) for a in alz_M],
            "Alzada Hembras (m)": [round(a,3) for a in alz_H],
        })
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 4: ACERCA DEL SISTEMA
# ══════════════════════════════════════════════════════════════

elif seccion == "ℹ️ Acerca del Sistema":

    st.markdown("""
    <div class="section-header">
        <h2>ℹ️ Acerca de CreciPSI</h2>
        <p>Metodología, validación y referencias del sistema</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Origen de los datos")
        st.markdown("""
        Los datos provienen de registros zootécnicos reales de un rancho 
        productor de Pura Sangre Inglés en México, recopilados durante 
        **10 años consecutivos (2015–2025)**. El dataset incluye 217 animales 
        con 4,175 mediciones de peso y 3,981 de alzada.
        """)

        st.markdown("#### Metodología estadística")
        st.markdown("""
        - **Curvas de referencia:** Percentiles P10, P25, P50, P75, P90 
          calculados por edad (0–22 meses) y sexo
        - **Modelo predictivo:** Regresión polinomial grado 3 con variables 
          sexo, edad y alzada
        - **Validación:** División train/test 80%/20% con semilla aleatoria fija
        - **Clasificador:** Basado en criterios clínicos equinos para 
          detección de patrones anómalos
        """)

    with col_b:
        st.markdown("#### Métricas de validación")

        metricas_tabla = pd.DataFrame({
            "Modelo": ["Peso (con alzada)", "Peso (sin alzada)", "Alzada"],
            "R²":     ["0.9641",            "0.9458",            "0.9552"],
            "MAE":    ["15.1 kg",           "19.6 kg",           "2.0 cm"],
            "N datos":["3,181",             "3,181",             "3,181"],
        })
        st.dataframe(metricas_tabla, use_container_width=True, hide_index=True)

        st.markdown("#### Limitaciones")
        st.markdown("""
        - Las curvas son específicas para esta población y rancho
        - No generalizable sin validación externa
        - El 10.6% de animales no tiene peso al nacer (principalmente 2015)
        - El modelo no incluye variables de alimentación o sanidad
        """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Referencias principales")
    st.markdown("""
    1. Hintz HF et al. Growth rate of thoroughbreds. *J Anim Sci.* 1979;48(3):480-487.
    2. National Research Council. *Nutrient Requirements of Horses.* 6th ed. 2007.
    3. WHO Multicentre Growth Reference Study Group. *WHO child growth standards.* 2006.
    4. James G et al. *An Introduction to Statistical Learning.* 2nd ed. Springer; 2021.
    5. Dohoo I et al. *Veterinary Epidemiologic Research.* 2nd ed. VER Inc; 2009.
    """)

    st.markdown("""
    <div class="info-box">
        <strong>Citación sugerida:</strong> [Tu nombre]. CreciPSI: Sistema de 
        monitoreo inteligente de crecimiento en potros Pura Sangre Inglés mediante 
        inteligencia artificial. FMVZ-UNAM. Diplomado en IA en Salud Global. 2026.
    </div>
    """, unsafe_allow_html=True)
