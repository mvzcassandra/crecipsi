# # ══════════════════════════════════════════════════════════════
# CreciPSI v4.0 — Una página, entrada rápida
# FMVZ-UNAM | Diplomado IA en Salud Global 2025-2026
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    #MainMenu, footer, .stDeployButton {display:none}
    html, body, [class*="css"] {
        font-family: 'Inter','Segoe UI',sans-serif;
    }
    .header {
        background: linear-gradient(135deg,#0f3460,#16213e);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .header h1 { font-size:2rem; font-weight:800; margin:0; }
    .header p  { font-size:0.9rem; opacity:0.8; margin:0.3rem 0 0 0; }
    .badge {
        display:inline-block;
        background:rgba(255,255,255,0.15);
        padding:0.2rem 0.8rem;
        border-radius:20px;
        font-size:0.75rem;
        margin-top:0.6rem;
    }
    .metric-row {
        display:flex; gap:1rem; margin-bottom:1.5rem;
    }
    .metric-card {
        flex:1; background:#f8fafc;
        border:1px solid #e2e8f0;
        border-radius:10px;
        padding:0.9rem 1rem;
        text-align:center;
    }
    .metric-card .val {
        font-size:1.6rem; font-weight:700; color:#0f3460;
    }
    .metric-card .lbl {
        font-size:0.72rem; color:#64748b;
        text-transform:uppercase; letter-spacing:0.4px;
    }
    .seccion-titulo {
        font-size:1.1rem; font-weight:700;
        color:#1e293b; margin:1.5rem 0 0.5rem 0;
        border-left:4px solid #0f3460;
        padding-left:0.7rem;
    }
    .patron-normal    {background:#dcfce7;color:#166534;border:1.5px solid #86efac;}
    .patron-superior  {background:#dbeafe;color:#1e3a8a;border:1.5px solid #93c5fd;}
    .patron-inferior  {background:#ffedd5;color:#9a3412;border:1.5px solid #fdba74;}
    .patron-irregular {background:#fee2e2;color:#991b1b;border:1.5px solid #fca5a5;}
    .patron-box {
        border-radius:8px; padding:0.8rem 1rem;
        font-size:1rem; font-weight:600;
        text-align:center; margin:0.5rem 0;
    }
    .info-box {
        background:#f0f9ff; border:1px solid #bae6fd;
        border-radius:8px; padding:0.8rem 1rem;
        font-size:0.85rem; color:#0c4a6e; margin:0.6rem 0;
    }
    .warn-box {
        background:#fffbeb; border:1px solid #fde68a;
        border-radius:8px; padding:0.8rem 1rem;
        font-size:0.85rem; color:#92400e; margin:0.6rem 0;
    }
    .ok-box {
        background:#f0fdf4; border:1px solid #bbf7d0;
        border-radius:8px; padding:0.8rem 1rem;
        font-size:0.85rem; color:#14532d; margin:0.6rem 0;
    }
    div[data-testid="stDataEditor"] table {font-size:0.9rem !important;}
    .stTabs [data-baseweb="tab"] {font-size:0.9rem; font-weight:600;}
    hr {border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0;}
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

COLORES = {
    "M":"#0f3460","H":"#831843",
    "normal":"#16a34a","superior":"#1d4ed8",
    "inferior":"#ea580c","irregular":"#dc2626",
}


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🐴 CreciPSI</h1>
    <p>Monitor Inteligente de Crecimiento en Potros Pura Sangre Inglés</p>
    <span class="badge">FMVZ-UNAM · Diplomado IA en Salud Global · 2025–2026</span>
</div>
""", unsafe_allow_html=True)

# Métricas globales
st.markdown("""
<div class="metric-row">
    <div class="metric-card"><div class="val">217</div><div class="lbl">Potros PSI</div></div>
    <div class="metric-card"><div class="val">4,175</div><div class="lbl">Mediciones</div></div>
    <div class="metric-card"><div class="val">10 años</div><div class="lbl">2015–2025</div></div>
    <div class="metric-card"><div class="val">0.964</div><div class="lbl">R² del modelo</div></div>
    <div class="metric-card"><div class="val">15 kg</div><div class="lbl">Error promedio</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS DE NAVEGACIÓN
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Curvas de Referencia",
    "🔍 Evaluar un Potro",
    "🎯 Predictor de Peso",
    "ℹ️ Metodología"
])


# ══════════════════════════════════════════════════════════════
# TAB 1: CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="seccion-titulo">Curvas percentiladas del rancho</p>',
                unsafe_allow_html=True)

    col_sx, col_var, _ = st.columns([1, 1, 3])
    with col_sx:
        sexo_c = st.radio("Sexo:", ["Machos ♂","Hembras ♀"],
                          horizontal=True, key="sc")
    with col_var:
        var_c = st.radio("Variable:", ["Peso (kg)","Alzada (m)"],
                         horizontal=True, key="vc")

    sk  = "M" if "Machos" in sexo_c else "H"
    col = COLORES[sk]
    st_data = stats_ref[f"stats_{sk}"] if "Peso" in var_c \
              else stats_alz[f"stats_{sk}"]
    campo   = "p50"
    unidad  = "kg" if "Peso" in var_c else "m"
    ylim    = (20, 570) if "Peso" in var_c else (0.85, 1.68)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    edades = st_data["edad_meses"]
    ax.fill_between(edades, st_data.p10, st_data.p90,
                    alpha=0.08, color=col)
    ax.fill_between(edades, st_data.p25, st_data.p75,
                    alpha=0.20, color=col, label="Rango normal (P25–P75)")
    ax.plot(edades, st_data.p50, color=col,
            linewidth=2.5, label="Mediana (P50)")
    ax.plot(edades, st_data.p10, color=col,
            linewidth=0.8, linestyle=":", alpha=0.5, label="P10 / P90")
    ax.plot(edades, st_data.p90, color=col,
            linewidth=0.8, linestyle=":", alpha=0.5)

    for mes_r in ([0,6,12,18] if "Peso" in var_c else [6,12,18]):
        f = st_data[st_data.edad_meses == mes_r]
        if len(f) > 0:
            v   = f["p50"].values[0]
            txt = f"{v:.0f} kg" if unidad=="kg" else f"{v:.2f} m"
            off = 18 if unidad=="kg" else 0.01
            ax.annotate(txt, xy=(mes_r,v),
                        xytext=(mes_r+0.5, v+off),
                        fontsize=8.5, color=col, fontweight="600",
                        arrowprops=dict(arrowstyle="->",color=col,lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white",ec=col,alpha=0.85))

    ax.set_xlabel("Edad (meses)", fontsize=11)
    ax.set_ylabel(f"{'Peso' if unidad=='kg' else 'Alzada'} ({unidad})", fontsize=11)
    n = 111 if sk=="M" else 106
    ax.set_title(
        f"Curvas de {'Peso' if unidad=='kg' else 'Alzada'} — "
        f"{'Machos' if sk=='M' else 'Hembras'} PSI  (n={n} animales)",
        fontsize=12, fontweight="700"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5 if "Peso" in var_c else 1, 22.3)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver tabla de valores"):
        t = st_data[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
        t.columns = ["Edad","P10","P25","P50","P75","P90","N"]
        st.dataframe(t.round(1 if unidad=="kg" else 3),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p class="seccion-titulo">Datos del potro</p>',
                unsafe_allow_html=True)

    # Datos básicos en una línea
    ci1, ci2, ci3 = st.columns([2, 1, 1])
    with ci1:
        nombre_e = st.text_input("Nombre / identificador",
                                 placeholder="Ej. Hijo de Mila Race",
                                 key="nombre_eval")
    with ci2:
        sexo_e = st.radio("Sexo:", ["Macho ♂","Hembra ♀"],
                          horizontal=True, key="sexo_eval")
    with ci3:
        peso_nac_e = st.number_input("Peso al nacer (kg)",
                                     min_value=0.0, max_value=80.0,
                                     value=0.0, step=0.5, key="pnac")

    sk_e = "M" if "Macho" in sexo_e else "H"

    # ── Tabla de entrada estilo Excel ────────────────────────
    st.markdown('<p class="seccion-titulo">Mediciones mensuales</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Escribe directamente en la tabla. Deja en <strong>0</strong> 
        los meses sin medición. La tabla se puede editar celda por celda.
    </div>
    """, unsafe_allow_html=True)

    # Crear tabla inicial con 22 meses
    df_entrada = pd.DataFrame({
        "Mes":       list(range(1, 23)),
        "Peso (kg)": [0.0] * 22,
        "Alzada (m)":[0.0] * 22,
    })

    # st.data_editor permite editar la tabla directamente
    df_editado = st.data_editor(
        df_entrada,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Mes": st.column_config.NumberColumn(
                "Mes", disabled=True, width="small"
            ),
            "Peso (kg)": st.column_config.NumberColumn(
                "Peso (kg)", min_value=0, max_value=700,
                step=0.5, format="%.1f"
            ),
            "Alzada (m)": st.column_config.NumberColumn(
                "Alzada (m)", min_value=0, max_value=2,
                step=0.01, format="%.3f"
            ),
        },
        key="tabla_mediciones"
    )

    # Botón de análisis
    st.markdown("")
    analizar = st.button("🔍 Analizar crecimiento", type="primary",
                         use_container_width=True, key="btn_analizar")

    if analizar:
        # Extraer mediciones con valor > 0
        pesos_e   = {}
        alzadas_e = {}

        if peso_nac_e > 0:
            pesos_e[0] = peso_nac_e

        for _, fila in df_editado.iterrows():
            mes = int(fila["Mes"])
            p   = float(fila["Peso (kg)"])
            a   = float(fila["Alzada (m)"])
            if p > 0: pesos_e[mes]   = p
            if a > 0: alzadas_e[mes] = a

        if len(pesos_e) < 2 and len(alzadas_e) < 2:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Ingresa al menos <strong>2 mediciones</strong> 
                de peso o alzada para analizar.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"### Resultados — {nombre_e or 'Potro evaluado'}")

        sp = stats_ref[f"stats_{sk_e}"]
        sa = stats_alz[f"stats_{sk_e}"]

        # ── Función de evaluación ─────────────────────────────
        def evaluar(mediciones, stats_v, campo):
            filas = []
            for edad in sorted(mediciones.keys()):
                v   = mediciones[edad]
                ref = stats_v[stats_v.edad_meses == int(edad)]
                if ref.empty: continue
                p10 = ref["p10"].values[0]
                p25 = ref["p25"].values[0]
                p50 = ref["p50"].values[0]
                p75 = ref["p75"].values[0]
                p90 = ref["p90"].values[0]
                diff = ((v - p50) / p50) * 100

                if v < p10:     zona="MUY BAJO"; alerta=True
                elif v < p25:   zona="BAJO";     alerta=True
                elif v <= p75:  zona="NORMAL";   alerta=False
                elif v <= p90:  zona="ALTO";     alerta=False
                else:           zona="MUY ALTO"; alerta=True

                filas.append({
                    "edad":edad, campo:round(v,3 if campo=="alzada" else 1),
                    "P25":round(p25,3 if campo=="alzada" else 1),
                    "P50":round(p50,3 if campo=="alzada" else 1),
                    "P75":round(p75,3 if campo=="alzada" else 1),
                    "diff":diff, "zona":zona, "alerta":alerta
                })
            return pd.DataFrame(filas)

        def patron(df, campo):
            if len(df) < 2: return "Sin datos","neutro"
            pa = df["zona"].str.contains("ALTO").mean()
            pb = df["zona"].str.contains("BAJO").mean()
            vals = list(df[campo])
            perd = sum(1 for i in range(1,len(vals))
                       if vals[i] < vals[i-1])
            caida = any((vals[i]-vals[i-1])/vals[i-1]*100 < -8
                        for i in range(1,len(vals)))
            if (perd >= 4) or caida: return "Irregular","irregular"
            elif pa >= 0.60: return "Superior","superior"
            elif pb >= 0.60: return "Inferior","inferior"
            return "Normal","normal"

        df_p = evaluar(pesos_e,   sp, "peso")
        df_a = evaluar(alzadas_e, sa, "alzada")

        pat_p, cls_p = patron(df_p, "peso")
        pat_a, cls_a = patron(df_a, "alzada")

        # ── Resultados compactos ──────────────────────────────
        rcol1, rcol2 = st.columns(2)

        with rcol1:
            if len(df_p) >= 2:
                st.markdown(f"""
                <div class="patron-box patron-{cls_p}">
                    Peso — Patrón {pat_p}
                </div>
                """, unsafe_allow_html=True)
                n_ap = df_p["alerta"].sum()
                if n_ap == 0:
                    st.markdown('<div class="ok-box">✅ Sin alertas de peso</div>',
                                unsafe_allow_html=True)
                else:
                    ma_p = df_p[df_p["alerta"]==True]["edad"].tolist()
                    st.markdown(f'<div class="warn-box">⚠️ Alertas en meses: {ma_p}</div>',
                                unsafe_allow_html=True)

        with rcol2:
            if len(df_a) >= 2:
                st.markdown(f"""
                <div class="patron-box patron-{cls_a}">
                    Alzada — Patrón {pat_a}
                </div>
                """, unsafe_allow_html=True)
                n_aa = df_a["alerta"].sum()
                if n_aa == 0:
                    st.markdown('<div class="ok-box">✅ Sin alertas de alzada</div>',
                                unsafe_allow_html=True)
                else:
                    ma_a = df_a[df_a["alerta"]==True]["edad"].tolist()
                    st.markdown(f'<div class="warn-box">⚠️ Alertas en meses: {ma_a}</div>',
                                unsafe_allow_html=True)

        # ── Gráficas ──────────────────────────────────────────
        n_g = (1 if len(df_p)>=1 else 0) + (1 if len(df_a)>=1 else 0)
        if n_g > 0:
            gcols = st.columns(n_g)
            gi    = 0

            for df_v, campo, stats_v, clr, ylabel in [
                (df_p,"peso",  sp, COLORES.get(cls_p,COLORES[sk_e]),"Peso (kg)"),
                (df_a,"alzada",sa, COLORES.get(cls_a,COLORES[sk_e]),"Alzada (m)"),
            ]:
                if len(df_v) < 1: continue
                with gcols[gi]:
                    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
                    fig2.patch.set_facecolor("#fafafa")
                    ax2.set_facecolor("#fafafa")

                    cref = COLORES[sk_e]
                    ax2.fill_between(stats_v.edad_meses,
                                     stats_v.p25, stats_v.p75,
                                     alpha=0.18, color=cref,
                                     label="P25–P75")
                    ax2.fill_between(stats_v.edad_meses,
                                     stats_v.p10, stats_v.p90,
                                     alpha=0.07, color=cref)
                    ax2.plot(stats_v.edad_meses, stats_v.p50,
                             color=cref, linewidth=1.8,
                             linestyle="--", alpha=0.6,
                             label="Mediana rancho")
                    ax2.plot(df_v["edad"], df_v[campo],
                             color=clr, linewidth=2.5,
                             marker="o", markersize=6,
                             label=nombre_e or "Potro", zorder=5)

                    alt = df_v[df_v["alerta"]==True]
                    if len(alt) > 0:
                        ax2.scatter(alt["edad"], alt[campo],
                                    color="#ef4444", s=100,
                                    marker="x", linewidths=2.5,
                                    zorder=6, label="Alerta")

                    ax2.set_xlabel("Edad (meses)", fontsize=10)
                    ax2.set_ylabel(ylabel, fontsize=10)
                    ax2.set_title(ylabel, fontsize=11, fontweight="700")
                    ax2.legend(fontsize=8, framealpha=0.9)
                    ax2.grid(True, alpha=0.18, linestyle="--")
                    ax2.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                gi += 1

        # ── Tabla resumen ─────────────────────────────────────
        if len(df_p) >= 1:
            with st.expander("Ver detalle de peso por mes"):
                ds = df_p[["edad","peso","P25","P50","P75","zona"]].copy()
                ds["vs P50"] = df_p["diff"].apply(lambda x: f"{x:+.1f}%")
                ds.columns   = ["Mes","Peso(kg)","P25","P50","P75","Estado","vs P50"]

                def color_e(v):
                    if "BAJO" in str(v):   return "background:#ffedd5;color:#9a3412"
                    elif "ALTO" in str(v): return "background:#dbeafe;color:#1e3a8a"
                    elif "NORMAL" in str(v):return "background:#dcfce7;color:#166534"
                    return ""

                st.dataframe(
                    ds.style.applymap(color_e, subset=["Estado"]),
                    use_container_width=True, hide_index=True
                )

        if len(df_a) >= 1:
            with st.expander("Ver detalle de alzada por mes"):
                ds2 = df_a[["edad","alzada","P25","P50","P75","zona"]].copy()
                ds2["vs P50"] = df_a["diff"].apply(lambda x: f"{x:+.1f}%")
                ds2.columns   = ["Mes","Alzada(m)","P25","P50","P75","Estado","vs P50"]
                st.dataframe(
                    ds2.style.applymap(color_e, subset=["Estado"]),
                    use_container_width=True, hide_index=True
                )


# ══════════════════════════════════════════════════════════════
# TAB 3: PREDICTOR
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<p class="seccion-titulo">Predictor de Peso y Alzada</p>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    with pc1:
        sexo_pr = st.radio("Sexo:", ["Macho ♂","Hembra ♀"],
                           horizontal=True, key="sexo_pr")
    with pc2:
        edad_pr = st.slider("Edad (meses):", 1, 22, 6, key="edad_pr")
    with pc3:
        alz_pr  = st.number_input(
            "Alzada actual (m) — opcional, mejora la precisión",
            min_value=0.0, max_value=2.0, value=0.0,
            step=0.01, key="alz_pr"
        )

    sk_pr   = "M" if "Macho" in sexo_pr else "H"
    sbin_pr = 1 if sk_pr == "M" else 0

    sp_pr = stats_ref[f"stats_{sk_pr}"]
    sa_pr = stats_alz[f"stats_{sk_pr}"]
    ref_a = sa_pr[sa_pr.edad_meses == edad_pr]
    alz_m = ref_a["p50"].values[0] if len(ref_a) > 0 else 1.35
    alz_u = alz_pr if alz_pr > 0 else alz_m

    peso_pr = mod_peso.predict([[sbin_pr, edad_pr, alz_u]])[0]
    alz_pr2 = mod_alz.predict([[sbin_pr, edad_pr, peso_pr]])[0]

    ref_p = sp_pr[sp_pr.edad_meses == edad_pr]

    st.markdown("<hr>", unsafe_allow_html=True)

    if len(ref_p) > 0:
        p25r = ref_p["p25"].values[0]
        p50r = ref_p["p50"].values[0]
        p75r = ref_p["p75"].values[0]

        pos   = "normal" if p25r <= peso_pr <= p75r \
                else ("superior" if peso_pr > p75r else "inferior")
        texto = {"normal":"Dentro del rango normal",
                 "superior":"Por encima del rango normal",
                 "inferior":"Por debajo del rango normal"}[pos]
        color_res = COLORES[pos]

        mr1, mr2, mr3, mr4 = st.columns(4)
        with mr1:
            st.markdown(f"""
            <div style="background:{color_res};color:white;
                        border-radius:10px;padding:1rem;text-align:center">
                <div style="font-size:2rem;font-weight:800">{peso_pr:.0f} kg</div>
                <div style="font-size:0.78rem;opacity:0.9">
                    Peso predicho — mes {edad_pr}
                </div>
                <div style="font-size:0.72rem;margin-top:0.3rem;
                            background:rgba(255,255,255,0.2);
                            border-radius:6px;padding:0.15rem">
                    {texto}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with mr2:
            st.metric("P25 del rancho", f"{p25r:.0f} kg")
        with mr3:
            st.metric("P50 del rancho", f"{p50r:.0f} kg")
        with mr4:
            st.metric("P75 del rancho", f"{p75r:.0f} kg")

        st.markdown("")

        ref_a2 = sa_pr[sa_pr.edad_meses == edad_pr]
        if len(ref_a2) > 0:
            a50 = ref_a2["p50"].values[0]
            a25 = ref_a2["p25"].values[0]
            a75 = ref_a2["p75"].values[0]
            ar1, ar2, ar3, ar4 = st.columns(4)
            with ar1:
                st.metric("Alzada predicha", f"{alz_pr2:.3f} m")
            with ar2:
                st.metric("P25 alzada", f"{a25:.3f} m")
            with ar3:
                st.metric("P50 alzada", f"{a50:.3f} m")
            with ar4:
                st.metric("P75 alzada", f"{a75:.3f} m")

    # Gráfica
    edades_g = list(range(1,23))
    saM = stats_alz["stats_M"]; saH = stats_alz["stats_H"]
    spM = stats_ref["stats_M"]; spH = stats_ref["stats_H"]

    aM = [saM[saM.edad_meses==e]["p50"].values[0]
          if len(saM[saM.edad_meses==e])>0 else 1.35 for e in edades_g]
    aH = [saH[saH.edad_meses==e]["p50"].values[0]
          if len(saH[saH.edad_meses==e])>0 else 1.33 for e in edades_g]

    pM_g = [mod_peso.predict([[1,e,a]])[0] for e,a in zip(edades_g,aM)]
    pH_g = [mod_peso.predict([[0,e,a]])[0] for e,a in zip(edades_g,aH)]

    fig5, ax5 = plt.subplots(figsize=(12,4.5))
    fig5.patch.set_facecolor("#fafafa")
    ax5.set_facecolor("#fafafa")
    ax5.fill_between(spM.edad_meses, spM.p25, spM.p75,
                     alpha=0.12, color=COLORES["M"])
    ax5.fill_between(spH.edad_meses, spH.p25, spH.p75,
                     alpha=0.12, color=COLORES["H"])
    ax5.plot(edades_g, pM_g, color=COLORES["M"],
             linewidth=2.5, label="Machos")
    ax5.plot(edades_g, pH_g, color=COLORES["H"],
             linewidth=2.5, label="Hembras")
    ax5.axvline(x=edad_pr, color="#94a3b8",
                linestyle="--", linewidth=1.5, alpha=0.8)
    ax5.scatter([edad_pr],[peso_pr], color="#f59e0b",
                s=180, zorder=6, edgecolors="white",
                linewidths=2, label=f"Predicción: {peso_pr:.0f} kg")
    ax5.set_xlabel("Edad (meses)", fontsize=11)
    ax5.set_ylabel("Peso (kg)", fontsize=11)
    ax5.set_title("Predicciones vs. rangos normales del rancho",
                  fontsize=12, fontweight="700")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.18, linestyle="--")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    with st.expander("Ver tabla completa de predicciones"):
        st.dataframe(pd.DataFrame({
            "Mes":              edades_g,
            "Peso Machos(kg)":  [round(p) for p in pM_g],
            "Peso Hembras(kg)": [round(p) for p in pH_g],
            "Alzada Machos(m)": [round(a,3) for a in aM],
            "Alzada Hembras(m)":[round(a,3) for a in aH],
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: METODOLOGÍA
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<p class="seccion-titulo">Metodología y validación</p>',
                unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### Base de datos")
        st.markdown("""
        Registros zootécnicos reales de un rancho PSI mexicano 
        (2015–2025). Dataset de 217 animales con 4,175 mediciones 
        de peso y 3,981 de alzada. Completitud del 100% en alzada.
        """)
        st.markdown("#### Estadística aplicada")
        st.markdown("""
        - Percentiles P10–P90 por edad (0–22 meses) y sexo
        - Regresión polinomial grado 3
        - Validación train/test 80%/20%
        - Clasificador basado en criterios clínicos equinos
        """)

    with m2:
        st.markdown("#### Métricas de validación")
        st.dataframe(pd.DataFrame({
            "Modelo":   ["Peso (con alzada)","Peso (sin alzada)","Alzada"],
            "R²":       ["0.9641","0.9458","0.9552"],
            "MAE":      ["15.1 kg","19.6 kg","2.0 cm"],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### Limitaciones")
        st.markdown("""
        - Curvas específicas para esta población y rancho
        - Sin variables de alimentación o sanidad
        - 10.6% sin peso al nacer (principalmente 2015)
        """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Referencias")
    st.markdown("""
    1. Hintz HF et al. *J Anim Sci.* 1979;48(3):480-487.
    2. NRC. *Nutrient Requirements of Horses.* 6th ed. 2007.
    3. WHO MGRS Group. *WHO child growth standards.* 2006.
    4. James G et al. *An Introduction to Statistical Learning.* Springer; 2021.
    5. Dohoo I et al. *Veterinary Epidemiologic Research.* VER Inc; 2009.
    """)
