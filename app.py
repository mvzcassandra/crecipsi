# ══════════════════════════════════════════════════════════════
# CreciPSI — Monitor de Crecimiento en Potros PSI
# FMVZ-UNAM | Diplomado IA en Salud Global 2025-2026
# Versión 2.0 — Incluye curvas de alzada y modelo mejorado
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de página ──
st.set_page_config(
    page_title="CreciPSI - Monitor de Crecimiento Equino",
    page_icon="horse",
    layout="wide"
)

# ── Cargar modelos ──
@st.cache_resource
def cargar_modelos():
    with open("stats_ref_final.pkl", "rb") as f:
        stats_ref = pickle.load(f)
    with open("stats_alzada_final.pkl", "rb") as f:
        stats_alzada = pickle.load(f)
    with open("modelo_peso_v2.pkl", "rb") as f:
        modelo_peso = pickle.load(f)
    with open("modelo_alzada.pkl", "rb") as f:
        modelo_alzada = pickle.load(f)
    return stats_ref, stats_alzada, modelo_peso, modelo_alzada

try:
    stats_ref, stats_alzada, modelo_peso, modelo_alzada = cargar_modelos()
    modelos_ok = True
except FileNotFoundError as e:
    modelos_ok = False
    error_msg = str(e)

# ── Encabezado ──
st.title("CreciPSI")
st.markdown("### Monitor Inteligente de Crecimiento en Potros Pura Sangre Ingles")
st.markdown(
    "**FMVZ-UNAM** | Diplomado en Inteligencia Artificial en Salud Global | 2025-2026"
)
st.markdown("---")

if not modelos_ok:
    st.error(f"No se encontraron los archivos de modelos: {error_msg}")
    st.stop()

# ── Menu lateral ──
seccion = st.sidebar.radio(
    "Selecciona una seccion:",
    [
        "Curvas de Peso",
        "Curvas de Alzada",
        "Evaluar un Potro",
        "Predictor Inteligente",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Acerca del sistema**")
st.sidebar.markdown(
    "Curvas de referencia construidas con **217 potros PSI** "
    "de un rancho mexicano (2015-2025)."
)
st.sidebar.markdown("**Modelo de peso:** R2=0.9641")
st.sidebar.markdown("**Modelo de alzada:** R2=0.9552")
st.sidebar.markdown("**Rango valido:** 1-22 meses")


# ══════════════════════════════════════════════════════════════
# SECCION 1: CURVAS DE PESO
# ══════════════════════════════════════════════════════════════

if seccion == "Curvas de Peso":
    st.header("Curvas de Crecimiento de Peso")
    st.markdown(
        "Curvas percentiladas construidas con **4,175 mediciones** de "
        "**217 potros PSI** nacidos entre 2015 y 2025. "
        "Incluye peso al nacer (mes 0) para los 194 animales con ese registro."
    )

    sexo_sel = st.radio("Sexo:", ["Machos", "Hembras"], horizontal=True)
    sexo_key = "M" if sexo_sel == "Machos" else "H"
    stats    = stats_ref[f"stats_{sexo_key}"]
    color    = "#1565C0" if sexo_key == "M" else "#AD1457"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(stats.edad_meses, stats.p10, stats.p90,
                    alpha=0.10, color=color, label="P10-P90")
    ax.fill_between(stats.edad_meses, stats.p25, stats.p75,
                    alpha=0.25, color=color, label="Rango normal (P25-P75)")
    ax.plot(stats.edad_meses, stats.p50, color=color,
            linewidth=2.5, label="Mediana (P50)")
    ax.plot(stats.edad_meses, stats.p10, color=color,
            linewidth=1, linestyle=":", alpha=0.6)
    ax.plot(stats.edad_meses, stats.p90, color=color,
            linewidth=1, linestyle=":", alpha=0.6)

    for mes_ref in [0, 6, 12, 18]:
        fila = stats[stats.edad_meses == mes_ref]
        if len(fila) > 0:
            p50_val = fila["p50"].values[0]
            offset  = 5 if mes_ref == 0 else 20
            ax.annotate(
                f"{p50_val:.0f} kg",
                xy=(mes_ref, p50_val),
                xytext=(mes_ref + 0.4, p50_val + offset),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1)
            )

    n_anim = 111 if sexo_key == "M" else 106
    ax.set_title(
        f"Curvas de Peso - {sexo_sel} PSI\n"
        f"Rancho mexicano 2015-2025 (n={n_anim} animales)",
        fontsize=12
    )
    ax.set_xlabel("Edad (meses)", fontsize=12)
    ax.set_ylabel("Peso (kg)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 22)
    ax.set_ylim(20, 560)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Tabla de referencia")
    tabla = stats[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
    tabla.columns = ["Edad (meses)","P10","P25","P50","P75","P90","N"]
    st.dataframe(tabla.round(1), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# SECCION 2: CURVAS DE ALZADA
# ══════════════════════════════════════════════════════════════

elif seccion == "Curvas de Alzada":
    st.header("Curvas de Alzada de Referencia")
    st.markdown(
        "Curvas percentiladas de alzada construidas con **3,981 mediciones** "
        "de 217 potros PSI. Completitud del 100% en todos los años. "
        "La alzada se mide en metros desde el suelo hasta la cruz."
    )

    sexo_sel = st.radio("Sexo:", ["Machos", "Hembras"], horizontal=True)
    sexo_key = "M" if sexo_sel == "Machos" else "H"
    stats    = stats_alzada[f"stats_{sexo_key}"]
    color    = "#1565C0" if sexo_key == "M" else "#AD1457"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(stats.edad_meses, stats.p10, stats.p90,
                    alpha=0.10, color=color, label="P10-P90")
    ax.fill_between(stats.edad_meses, stats.p25, stats.p75,
                    alpha=0.25, color=color, label="Rango normal (P25-P75)")
    ax.plot(stats.edad_meses, stats.p50, color=color,
            linewidth=2.5, label="Mediana (P50)")
    ax.plot(stats.edad_meses, stats.p10, color=color,
            linewidth=1, linestyle=":", alpha=0.6)
    ax.plot(stats.edad_meses, stats.p90, color=color,
            linewidth=1, linestyle=":", alpha=0.6)

    for mes_ref in [6, 12, 18]:
        fila = stats[stats.edad_meses == mes_ref]
        if len(fila) > 0:
            p50_val = fila["p50"].values[0]
            ax.annotate(
                f"{p50_val:.2f} m",
                xy=(mes_ref, p50_val),
                xytext=(mes_ref + 0.4, p50_val + 0.012),
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1)
            )

    n_anim = 111 if sexo_key == "M" else 106
    ax.set_title(
        f"Curvas de Alzada - {sexo_sel} PSI\n"
        f"Rancho mexicano 2015-2025 (n={n_anim} animales)",
        fontsize=12
    )
    ax.set_xlabel("Edad (meses)", fontsize=12)
    ax.set_ylabel("Alzada (metros)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 22)
    ax.set_ylim(0.85, 1.65)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Tabla de referencia de alzada")
    tabla = stats[["edad_meses","p10","p25","p50","p75","p90","n"]].copy()
    tabla.columns = ["Edad (meses)","P10","P25","P50","P75","P90","N"]
    st.dataframe(tabla.round(3), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Hallazgo clinico relevante")
    st.info(
        "La alzada presenta una correlacion de Pearson r=0.9666 con el peso "
        "corporal (p<0.001). Esto indica que el crecimiento en altura y en "
        "masa corporal son procesos altamente coordinados en esta poblacion PSI. "
        "Un potro con alzada normal para su edad pero peso bajo debe evaluarse "
        "para descartar deficit nutricional sin compromiso esqueletico."
    )


# ══════════════════════════════════════════════════════════════
# SECCION 3: EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════

elif seccion == "Evaluar un Potro":
    st.header("Evaluacion de Crecimiento Individual")
    st.markdown(
        "Ingresa los datos del potro para comparar su crecimiento "
        "contra las curvas de referencia del rancho. "
        "Puedes ingresar peso, alzada, o ambos."
    )

    col_entrada, col_resultado = st.columns([1, 2])

    with col_entrada:
        st.subheader("Datos del potro")
        nombre   = st.text_input("Nombre o identificador", value="Potro evaluado")
        sexo_sel = st.radio("Sexo:", ["Macho", "Hembra"], horizontal=True)
        sexo_key = "M" if "Macho" in sexo_sel else "H"

        st.markdown("**Mediciones por mes:**")
        st.caption("Ingresa 0 si no hay medicion en ese mes")

        pesos_input   = {}
        alzadas_input = {}

        for mes in range(0, 23):
            c1, c2 = st.columns(2)
            with c1:
                etiqueta = f"Mes {mes} peso(kg)" if mes > 0 else "Nac. peso(kg)"
                peso_val = st.number_input(
                    etiqueta, min_value=0.0, max_value=700.0,
                    value=0.0, step=1.0, key=f"p_{mes}"
                )
                if peso_val > 0:
                    pesos_input[mes] = peso_val
            with c2:
                if mes > 0:
                    alz_val = st.number_input(
                        f"Mes {mes} alzada(m)",
                        min_value=0.0, max_value=2.0,
                        value=0.0, step=0.01, key=f"a_{mes}"
                    )
                    if alz_val > 0:
                        alzadas_input[mes] = alz_val

    with col_resultado:
        if len(pesos_input) < 2 and len(alzadas_input) < 2:
            st.info("Ingresa al menos 2 mediciones para ver la evaluacion.")
        else:
            stats_p = stats_ref[f"stats_{sexo_key}"]
            stats_a = stats_alzada[f"stats_{sexo_key}"]
            color_ref = "#1565C0" if sexo_key == "M" else "#AD1457"

            # ── Evaluar peso ──
            filas_peso = []
            for edad in sorted(pesos_input.keys()):
                peso = pesos_input[edad]
                ref  = stats_p[stats_p.edad_meses == int(edad)]
                if ref.empty:
                    continue
                p10 = ref["p10"].values[0]
                p25 = ref["p25"].values[0]
                p50 = ref["p50"].values[0]
                p75 = ref["p75"].values[0]
                p90 = ref["p90"].values[0]
                diff = ((peso - p50) / p50) * 100

                if peso < p10:
                    zona = "MUY BAJO"; alerta = True
                elif peso < p25:
                    zona = "BAJO"; alerta = True
                elif peso <= p75:
                    zona = "NORMAL"; alerta = False
                elif peso <= p90:
                    zona = "ALTO"; alerta = False
                else:
                    zona = "MUY ALTO"; alerta = True

                filas_peso.append({
                    "edad_meses": edad, "peso_kg": peso,
                    "P25": round(p25,1), "P50": round(p50,1),
                    "P75": round(p75,1),
                    "vs_mediana": f"{diff:+.1f}%",
                    "zona_peso": zona, "alerta_peso": alerta,
                })

            # ── Evaluar alzada ──
            filas_alzada = []
            for edad in sorted(alzadas_input.keys()):
                alzada = alzadas_input[edad]
                ref    = stats_a[stats_a.edad_meses == int(edad)]
                if ref.empty:
                    continue
                p25 = ref["p25"].values[0]
                p50 = ref["p50"].values[0]
                p75 = ref["p75"].values[0]
                diff = ((alzada - p50) / p50) * 100

                if alzada < ref["p10"].values[0]:
                    zona = "MUY BAJA"; alerta = True
                elif alzada < p25:
                    zona = "BAJA"; alerta = True
                elif alzada <= p75:
                    zona = "NORMAL"; alerta = False
                elif alzada <= ref["p90"].values[0]:
                    zona = "ALTA"; alerta = False
                else:
                    zona = "MUY ALTA"; alerta = True

                filas_alzada.append({
                    "edad_meses": edad, "alzada_m": alzada,
                    "P25_alz": round(p25,3), "P50_alz": round(p50,3),
                    "P75_alz": round(p75,3),
                    "vs_mediana_alz": f"{diff:+.1f}%",
                    "zona_alzada": zona, "alerta_alzada": alerta,
                })

            df_peso   = pd.DataFrame(filas_peso)
            df_alzada = pd.DataFrame(filas_alzada)

            # ── Patron de peso ──
            if len(df_peso) >= 2:
                prop_alto = df_peso["zona_peso"].str.contains("ALTO").mean()
                prop_bajo = df_peso["zona_peso"].str.contains("BAJO").mean()

                pesos_ord = [(e, pesos_input[e])
                             for e in sorted(pesos_input.keys())]
                perdidas = sum(
                    1 for i in range(1, len(pesos_ord))
                    if pesos_ord[i][1] < pesos_ord[i-1][1]
                    and (pesos_ord[i][0] - pesos_ord[i-1][0]) <= 3
                )
                caida = any(
                    (pesos_ord[i][1]-pesos_ord[i-1][1])/pesos_ord[i-1][1]*100 < -8
                    for i in range(1, len(pesos_ord))
                    if (pesos_ord[i][0]-pesos_ord[i-1][0]) <= 3
                )

                if (perdidas >= 4) or caida:
                    patron_peso = "Irregular"
                    color_patron = "#B71C1C"
                elif prop_alto >= 0.60:
                    patron_peso = "Superior"
                    color_patron = "#1565C0"
                elif prop_bajo >= 0.60:
                    patron_peso = "Inferior"
                    color_patron = "#E65100"
                else:
                    patron_peso = "Normal"
                    color_patron = "#2E7D32"

                st.markdown(
                    f"<div style='background-color:{color_patron};"
                    f"padding:12px;border-radius:8px;color:white;"
                    f"font-size:20px;font-weight:bold;text-align:center'>"
                    f"Patron de peso: {patron_peso}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("")

                n_alertas_peso = df_peso["alerta_peso"].sum()
                if n_alertas_peso == 0:
                    st.success("Sin alertas de peso")
                else:
                    meses_alerta = df_peso[
                        df_peso["alerta_peso"] == True
                    ]["edad_meses"].tolist()
                    st.warning(f"Alertas de peso en meses: {meses_alerta}")

            # ── Patron de alzada ──
            if len(df_alzada) >= 2:
                prop_bajo_alz = df_alzada["zona_alzada"].str.contains("BAJA").mean()
                prop_alto_alz = df_alzada["zona_alzada"].str.contains("ALTA").mean()

                if prop_bajo_alz >= 0.60:
                    patron_alz = "Alzada Inferior"
                    color_alz  = "#E65100"
                elif prop_alto_alz >= 0.60:
                    patron_alz = "Alzada Superior"
                    color_alz  = "#1565C0"
                else:
                    patron_alz = "Alzada Normal"
                    color_alz  = "#2E7D32"

                st.markdown(
                    f"<div style='background-color:{color_alz};"
                    f"padding:12px;border-radius:8px;color:white;"
                    f"font-size:20px;font-weight:bold;text-align:center;"
                    f"margin-top:10px'>"
                    f"Patron de alzada: {patron_alz}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("")

                n_alertas_alz = df_alzada["alerta_alzada"].sum()
                if n_alertas_alz == 0:
                    st.success("Sin alertas de alzada")
                else:
                    meses_alz = df_alzada[
                        df_alzada["alerta_alzada"] == True
                    ]["edad_meses"].tolist()
                    st.warning(f"Alertas de alzada en meses: {meses_alz}")

            # ── Grafica combinada ──
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

            # Grafica de peso
            ax_p = axes2[0]
            ax_p.fill_between(stats_p.edad_meses, stats_p.p25, stats_p.p75,
                              alpha=0.20, color=color_ref, label="P25-P75")
            ax_p.fill_between(stats_p.edad_meses, stats_p.p10, stats_p.p90,
                              alpha=0.08, color=color_ref)
            ax_p.plot(stats_p.edad_meses, stats_p.p50, color=color_ref,
                     linewidth=2, linestyle="--", alpha=0.7, label="Mediana")
            if len(df_peso) >= 1:
                ax_p.plot(df_peso["edad_meses"], df_peso["peso_kg"],
                         color=color_patron, linewidth=2.5,
                         marker="o", markersize=7,
                         label=nombre, zorder=5)
                alertas_p = df_peso[df_peso["alerta_peso"] == True]
                if len(alertas_p) > 0:
                    ax_p.scatter(alertas_p["edad_meses"], alertas_p["peso_kg"],
                                color="red", s=120, marker="x",
                                linewidths=2.5, zorder=6, label="Alerta")
            ax_p.set_xlabel("Edad (meses)")
            ax_p.set_ylabel("Peso (kg)")
            ax_p.set_title(f"Peso — {nombre}")
            ax_p.legend(fontsize=8)
            ax_p.grid(True, alpha=0.3)

            # Grafica de alzada
            ax_a = axes2[1]
            ax_a.fill_between(stats_a.edad_meses, stats_a.p25, stats_a.p75,
                              alpha=0.20, color=color_ref, label="P25-P75")
            ax_a.fill_between(stats_a.edad_meses, stats_a.p10, stats_a.p90,
                              alpha=0.08, color=color_ref)
            ax_a.plot(stats_a.edad_meses, stats_a.p50, color=color_ref,
                     linewidth=2, linestyle="--", alpha=0.7, label="Mediana")
            if len(df_alzada) >= 1:
                color_alz2 = color_alz if len(df_alzada) >= 2 else color_ref
                ax_a.plot(df_alzada["edad_meses"], df_alzada["alzada_m"],
                         color=color_alz2, linewidth=2.5,
                         marker="s", markersize=7,
                         label=nombre, zorder=5)
                alertas_a = df_alzada[df_alzada["alerta_alzada"] == True]
                if len(alertas_a) > 0:
                    ax_a.scatter(alertas_a["edad_meses"], alertas_a["alzada_m"],
                                color="red", s=120, marker="x",
                                linewidths=2.5, zorder=6, label="Alerta")
            ax_a.set_xlabel("Edad (meses)")
            ax_a.set_ylabel("Alzada (m)")
            ax_a.set_title(f"Alzada — {nombre}")
            ax_a.legend(fontsize=8)
            ax_a.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # ── Tablas detalladas ──
            if len(df_peso) >= 1:
                st.markdown("---")
                st.subheader("Evaluacion de peso mes a mes")
                cols_show = ["edad_meses","peso_kg","P25","P50","P75",
                             "vs_mediana","zona_peso"]
                df_show = df_peso[cols_show].copy()
                df_show.columns = ["Edad","Peso(kg)","P25","P50",
                                   "P75","vs Mediana","Estado"]
                def color_zona(v):
                    if "BAJO" in str(v):
                        return "background-color: #FFCCBC"
                    elif "ALTO" in str(v):
                        return "background-color: #BBDEFB"
                    elif "NORMAL" in str(v):
                        return "background-color: #C8E6C9"
                    return ""
                st.dataframe(
                    df_show.style.applymap(color_zona, subset=["Estado"]),
                    use_container_width=True, hide_index=True
                )

            if len(df_alzada) >= 1:
                st.markdown("---")
                st.subheader("Evaluacion de alzada mes a mes")
                cols_alz = ["edad_meses","alzada_m","P25_alz","P50_alz",
                            "P75_alz","vs_mediana_alz","zona_alzada"]
                df_alz_show = df_alzada[cols_alz].copy()
                df_alz_show.columns = ["Edad","Alzada(m)","P25","P50",
                                       "P75","vs Mediana","Estado"]
                st.dataframe(
                    df_alz_show.style.applymap(color_zona, subset=["Estado"]),
                    use_container_width=True, hide_index=True
                )


# ══════════════════════════════════════════════════════════════
# SECCION 4: PREDICTOR INTELIGENTE
# ══════════════════════════════════════════════════════════════

elif seccion == "Predictor Inteligente":
    st.header("Predictor Inteligente de Peso y Alzada")
    st.markdown(
        "El predictor usa dos modelos segun los datos disponibles. "
        "Si ingresas la alzada usa el **modelo mejorado (R2=0.9641)**. "
        "Si solo tienes sexo y edad usa el **modelo base (R2=0.9458)**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datos de entrada")
        sexo_pred = st.radio("Sexo:", ["Macho", "Hembra"], horizontal=True)
        sexo_bin  = 1 if "Macho" in sexo_pred else 0
        sexo_key  = "M" if sexo_bin == 1 else "H"

        edad_pred = st.slider("Edad (meses):", 1, 22, 6)

        st.markdown("**Alzada actual (opcional):**")
        st.caption("Si la ingresas, el predictor sera mas preciso")
        alzada_input = st.number_input(
            "Alzada (metros):",
            min_value=0.0, max_value=2.0,
            value=0.0, step=0.01
        )

        usar_alzada = alzada_input > 0

        if usar_alzada:
            st.success("Usando modelo mejorado con alzada (R2=0.9641)")
        else:
            st.info("Usando modelo base sin alzada (R2=0.9458)")

    with col2:
        st.subheader("Predicciones")

        # Prediccion de peso
        if usar_alzada:
            peso_pred = modelo_peso.predict(
                [[sexo_bin, edad_pred, alzada_input]])[0]
            modelo_usado = "Modelo con alzada"
        else:
            # Usar alzada mediana del rancho para ese mes y sexo
            stats_a   = stats_alzada[f"stats_{sexo_key}"]
            ref_alz   = stats_a[stats_a.edad_meses == edad_pred]
            alzada_med = ref_alz["p50"].values[0] if len(ref_alz) > 0 else 1.35
            peso_pred  = modelo_peso.predict(
                [[sexo_bin, edad_pred, alzada_med]])[0]
            modelo_usado = "Modelo base"

        # Prediccion de alzada esperada dado el peso
        alzada_pred = modelo_alzada.predict(
            [[sexo_bin, edad_pred, peso_pred]])[0]

        # Percentiles de referencia
        stats_p   = stats_ref[f"stats_{sexo_key}"]
        stats_a   = stats_alzada[f"stats_{sexo_key}"]
        ref_peso  = stats_p[stats_p.edad_meses == edad_pred]
        ref_alz   = stats_a[stats_a.edad_meses == edad_pred]

        st.metric(
            label=f"Peso predicho a los {edad_pred} meses",
            value=f"{peso_pred:.0f} kg"
        )
        if len(ref_peso) > 0:
            p25p = ref_peso["p25"].values[0]
            p50p = ref_peso["p50"].values[0]
            p75p = ref_peso["p75"].values[0]
            st.markdown(f"**Mediana del rancho:** {p50p:.0f} kg")
            st.markdown(f"**Rango normal:** {p25p:.0f} - {p75p:.0f} kg")

        st.markdown("---")

        st.metric(
            label=f"Alzada esperada a los {edad_pred} meses",
            value=f"{alzada_pred:.3f} m"
        )
        if len(ref_alz) > 0:
            p25a = ref_alz["p25"].values[0]
            p50a = ref_alz["p50"].values[0]
            p75a = ref_alz["p75"].values[0]
            st.markdown(f"**Mediana del rancho:** {p50a:.3f} m")
            st.markdown(f"**Rango normal:** {p25a:.3f} - {p75a:.3f} m")

    # ── Grafica de curvas completas ──
    st.markdown("---")
    st.subheader("Curvas completas de prediccion")

    edades    = list(range(1, 23))
    stats_aM  = stats_alzada["stats_M"]
    stats_aH  = stats_alzada["stats_H"]

    alz_med_M = [stats_aM[stats_aM.edad_meses==e]["p50"].values[0]
                 if len(stats_aM[stats_aM.edad_meses==e]) > 0 else 1.35
                 for e in edades]
    alz_med_H = [stats_aH[stats_aH.edad_meses==e]["p50"].values[0]
                 if len(stats_aH[stats_aH.edad_meses==e]) > 0 else 1.33
                 for e in edades]

    preds_M = [modelo_peso.predict([[1, e, a]])[0]
               for e, a in zip(edades, alz_med_M)]
    preds_H = [modelo_peso.predict([[0, e, a]])[0]
               for e, a in zip(edades, alz_med_H)]

    stats_pM = stats_ref["stats_M"]
    stats_pH = stats_ref["stats_H"]

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

    # Grafica de peso
    ax3p = axes3[0]
    ax3p.fill_between(stats_pM.edad_meses, stats_pM.p25, stats_pM.p75,
                      alpha=0.12, color="#1565C0", label="Rango normal M")
    ax3p.fill_between(stats_pH.edad_meses, stats_pH.p25, stats_pH.p75,
                      alpha=0.12, color="#AD1457", label="Rango normal H")
    ax3p.plot(edades, preds_M, color="#1565C0",
              linewidth=2.5, label="Prediccion Machos")
    ax3p.plot(edades, preds_H, color="#AD1457",
              linewidth=2.5, label="Prediccion Hembras")
    ax3p.axvline(x=edad_pred, color="gray", linestyle="--", alpha=0.7)
    ax3p.scatter([edad_pred], [peso_pred], color="orange", s=200, zorder=5)
    ax3p.set_xlabel("Edad (meses)")
    ax3p.set_ylabel("Peso (kg)")
    ax3p.set_title("Prediccion de peso")
    ax3p.legend(fontsize=9)
    ax3p.grid(True, alpha=0.3)

    # Grafica de alzada
    ax3a = axes3[1]
    ax3a.fill_between(stats_aM.edad_meses, stats_aM.p25, stats_aM.p75,
                      alpha=0.12, color="#1565C0", label="Rango normal M")
    ax3a.fill_between(stats_aH.edad_meses, stats_aH.p25, stats_aH.p75,
                      alpha=0.12, color="#AD1457", label="Rango normal H")
    ax3a.plot(edades, alz_med_M, color="#1565C0",
              linewidth=2.5, label="Mediana Machos")
    ax3a.plot(edades, alz_med_H, color="#AD1457",
              linewidth=2.5, label="Mediana Hembras")
    ax3a.axvline(x=edad_pred, color="gray", linestyle="--", alpha=0.7)
    ax3a.scatter([edad_pred], [alzada_pred], color="orange", s=200, zorder=5)
    ax3a.set_xlabel("Edad (meses)")
    ax3a.set_ylabel("Alzada (m)")
    ax3a.set_title("Prediccion de alzada")
    ax3a.legend(fontsize=9)
    ax3a.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Tabla completa ──
    st.markdown("---")
    st.subheader("Tabla de predicciones completa")
    tabla_completa = pd.DataFrame({
        "Edad (meses)":          edades,
        "Peso Machos (kg)":      [round(p) for p in preds_M],
        "Peso Hembras (kg)":     [round(p) for p in preds_H],
        "Alzada Machos (m)":     [round(a, 3) for a in alz_med_M],
        "Alzada Hembras (m)":    [round(a, 3) for a in alz_med_H],
    })
    st.dataframe(tabla_completa, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Metricas de validacion")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("R2 modelo peso", "0.9641")
    with c2:
        st.metric("MAE peso", "15.1 kg")
    with c3:
        st.metric("R2 modelo alzada", "0.9552")
    with c4:
        st.metric("MAE alzada", "2.0 cm")
