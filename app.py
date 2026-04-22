# ══════════════════════════════════════════════════════════════════
# CreciPSI — Monitor de Crecimiento en Potros Pura Sangre Ingles
# FMVZ-UNAM | Diplomado IA en Salud Global 2025-2026
# ══════════════════════════════════════════════════════════════════

# ── IMPORTACIONES ──────────────────────────────────────────────────
# Streamlit es la librería que convierte este código en una app web
import streamlit as st

# Pandas maneja tablas de datos (DataFrames)
import pandas as pd

# Numpy hace cálculos matemáticos sobre arreglos de números
import numpy as np

# Matplotlib crea las gráficas
import matplotlib.pyplot as plt

# Pickle carga los modelos que entrenamos en Colab
import pickle

# Warnings silencia mensajes de advertencia no críticos
import warnings
warnings.filterwarnings("ignore")


# ── CONFIGURACIÓN DE LA PÁGINA ──────────────────────────────────────
# Esta es la primera instrucción que Streamlit ejecuta
# Configura el título que aparece en la pestaña del navegador,
# el ícono, y el layout (wide = usa todo el ancho de la pantalla)
st.set_page_config(
    page_title="CreciPSI - Monitor de Crecimiento Equino",
    page_icon="🐴",
    layout="wide"
)


# ── CARGAR MODELOS ──────────────────────────────────────────────────
# @st.cache_resource le dice a Streamlit que cargue estos archivos
# UNA SOLA VEZ y los guarde en memoria.
# Sin esto, cada vez que el usuario interactúa con la app
# (mueve un slider, escribe algo) Python recargaría los archivos
# desde cero — lo que haría la app muy lenta.
# cache_resource es para objetos pesados como modelos de ML.
@st.cache_resource
def cargar_modelos():
    # Abre el archivo pkl de estadísticas de referencia
    # 'rb' significa 'read binary' — los pkl son archivos binarios
    with open("stats_ref_final.pkl", "rb") as f:
        stats_ref = pickle.load(f)  # Carga el diccionario con P10-P90 por edad y sexo

    # Abre el modelo de regresión polinomial entrenado en Colab
    with open("modelo_final.pkl", "rb") as f:
        modelo = pickle.load(f)  # Carga el Pipeline de sklearn ya entrenado

    # Retorna ambos objetos para usarlos en toda la app
    return stats_ref, modelo

# Ejecutar la carga y guardar en variables globales
# Si falla (archivo no encontrado), muestra error y detiene la app
try:
    stats_ref, modelo = cargar_modelos()
    modelos_cargados = True
except FileNotFoundError:
    modelos_cargados = False


# ── ENCABEZADO PRINCIPAL ─────────────────────────────────────────────
# st.title muestra texto grande como título H1
st.title("🐴 CreciPSI")

# st.markdown permite usar formato Markdown para texto enriquecido
# ### = encabezado H3
st.markdown("### Monitor Inteligente de Crecimiento en Potros Pura Sangre Inglés")
st.markdown(
    "**FMVZ-UNAM** | Diplomado en Inteligencia Artificial en Salud Global | 2025–2026"
)

# st.markdown("---") dibuja una línea horizontal separadora
st.markdown("---")

# Si los modelos no cargaron, mostrar error y detener la ejecución
# st.error muestra una caja roja con el mensaje de error
# st.stop() detiene la app — no ejecuta nada más después de esto
if not modelos_cargados:
    st.error(
        "No se encontraron los archivos de modelos. "
        "Asegúrate de que stats_ref_final.pkl y modelo_final.pkl "
        "estén en la misma carpeta que app.py"
    )
    st.stop()


# ── MENÚ LATERAL ─────────────────────────────────────────────────────
# st.sidebar es el panel izquierdo de la app
# Todo lo que se escriba dentro de st.sidebar aparece ahí
# st.sidebar.radio crea botones de selección exclusiva (radio buttons)
# El usuario elige UNA de las opciones y eso cambia la sección visible
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Curvas de Referencia",
        "🔍 Evaluar un Potro",
        "🎯 Predictor de Peso"
    ],
    index=0  # La primera opción está seleccionada por defecto
)

# Información en el sidebar sobre el sistema
st.sidebar.markdown("---")
st.sidebar.markdown("**Acerca del sistema**")
st.sidebar.markdown(
    "Curvas de referencia construidas con **217 potros PSI** "
    "de un rancho mexicano (2015–2025). "
    "Modelo con **R²=0.93**."
)
st.sidebar.markdown("**Rango válido:** 1–22 meses")


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 1: CURVAS DE REFERENCIA
# ══════════════════════════════════════════════════════════════════
# La condición if compara el valor de 'seccion' con el string
# correspondiente. Solo se ejecuta el bloque cuya condición es True.
# Esto es lo que hace que la app muestre diferentes contenidos
# según lo que el usuario seleccionó en el menú.

if seccion == "📊 Curvas de Referencia":

    # Título de la sección
    st.header("Curvas de Crecimiento de Referencia")

    # Descripción metodológica — importante para el examen profesional
    st.markdown(
        "Curvas percentiladas construidas con **3,981 mediciones** de "
        "**217 potros PSI** nacidos entre 2015 y 2025 en un rancho mexicano. "
        "Metodología equivalente a las curvas de la OMS para crecimiento infantil, "
        "aplicada a équidos de esta población específica."
    )

    # st.radio horizontal para elegir sexo
    # horizontal=True pone los botones en línea en lugar de columna
    sexo_elegido = st.radio(
        "Selecciona sexo:",
        ["Machos ♂", "Hembras ♀"],
        horizontal=True
    )

    # Convertir la elección del usuario a la clave del diccionario
    # stats_ref tiene claves 'stats_M' y 'stats_H'
    sexo_key = "M" if "Machos" in sexo_elegido else "H"

    # Recuperar las estadísticas del sexo elegido
    # stats es un DataFrame con columnas: edad_meses, p10, p25, p50, p75, p90, n
    stats = stats_ref[f"stats_{sexo_key}"]

    # Color azul para machos, rosa para hembras
    color = "#1565C0" if sexo_key == "M" else "#AD1457"

    # ── Crear la gráfica con matplotlib ──────────────────────────────
    # figsize=(12, 6) define el tamaño en pulgadas: 12 de ancho, 6 de alto
    fig, ax = plt.subplots(figsize=(12, 6))

    # fill_between rellena el área entre dos curvas
    # alpha controla la transparencia (0=invisible, 1=sólido)
    # P10-P90: zona muy transparente = 80% de la población
    ax.fill_between(
        stats.edad_meses,   # eje X: edades de 1 a 22
        stats.p10,          # límite inferior del relleno
        stats.p90,          # límite superior del relleno
        alpha=0.10,         # muy transparente
        color=color,
        label="P10–P90 (80% de la población)"
    )

    # P25-P75: zona más opaca = rango normal (50% central)
    ax.fill_between(
        stats.edad_meses,
        stats.p25,
        stats.p75,
        alpha=0.25,         # más opaco que el anterior
        color=color,
        label="P25–P75 (rango normal)"
    )

    # Línea de la mediana P50 — la más gruesa y visible
    ax.plot(
        stats.edad_meses,
        stats.p50,
        color=color,
        linewidth=2.5,      # grosor de la línea
        label="Mediana (P50)"
    )

    # Líneas punteadas para P10 y P90
    ax.plot(stats.edad_meses, stats.p10, color=color,
            linewidth=1, linestyle=":", alpha=0.6)
    ax.plot(stats.edad_meses, stats.p90, color=color,
            linewidth=1, linestyle=":", alpha=0.6)

    # Anotar valores de referencia clínica en meses 6, 12 y 18
    # Estos son los momentos más importantes: destete (~6m),
    # un año (12m) y entrada a entrenamiento (~18m)
    for mes_clave in [6, 12, 18]:
        fila = stats[stats.edad_meses == mes_clave]
        if len(fila) > 0:
            valor_p50 = fila["p50"].values[0]
            # annotate dibuja una flecha con texto
            ax.annotate(
                f"{valor_p50:.0f} kg",           # texto de la anotación
                xy=(mes_clave, valor_p50),        # punto al que apunta la flecha
                xytext=(mes_clave + 0.4, valor_p50 + 20),  # posición del texto
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1)
            )

    # Configurar ejes y título
    ax.set_xlabel("Edad (meses)", fontsize=12)
    ax.set_ylabel("Peso (kg)", fontsize=12)
    n_animales = 111 if sexo_key == "M" else 106
    ax.set_title(
        f"Curvas de Crecimiento — {'Machos' if sexo_key=='M' else 'Hembras'} PSI\n"
        f"Rancho mexicano 2015–2025 (n={n_animales} animales)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 22)
    ax.set_ylim(50, 560)

    # st.pyplot muestra la figura de matplotlib en la app
    st.pyplot(fig)
    # Cerrar la figura para liberar memoria
    plt.close(fig)

    # ── Tabla de referencia ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Tabla de valores de referencia por edad")
    st.markdown(
        "Esta tabla es la base estadística de las curvas. "
        "Cada fila representa los percentiles de peso (kg) "
        "calculados con todos los animales de esa edad en el dataset."
    )

    # Preparar tabla para mostrar
    tabla_display = stats[[
        "edad_meses", "p10", "p25", "p50", "p75", "p90", "n"
    ]].copy()

    # Renombrar columnas para que sean más claras
    tabla_display.columns = [
        "Edad (meses)", "P10", "P25",
        "P50 (mediana)", "P75", "P90", "N animales"
    ]

    # Redondear a 1 decimal
    tabla_display = tabla_display.round(1)

    # st.dataframe muestra una tabla interactiva (ordenable)
    # use_container_width=True la hace del ancho total disponible
    # hide_index=True oculta el índice numérico de pandas
    st.dataframe(tabla_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 2: EVALUAR UN POTRO
# ══════════════════════════════════════════════════════════════════

elif seccion == "🔍 Evaluar un Potro":

    st.header("Evaluación de Crecimiento Individual")
    st.markdown(
        "Ingresa los pesos mensuales de un potro para comparar "
        "su curva de crecimiento contra las referencias del rancho. "
        "Deja en **0** los meses donde no hay medición."
    )

    # ── Layout de dos columnas ────────────────────────────────────────
    # st.columns([1, 2]) crea 2 columnas
    # El primer número define el ancho relativo de cada columna
    # [1, 2] = la segunda columna es el doble de ancha que la primera
    col_entrada, col_resultado = st.columns([1, 2])

    # ── Columna izquierda: entrada de datos ──────────────────────────
    with col_entrada:
        st.subheader("Datos del potro")

        # st.text_input crea un campo de texto
        nombre = st.text_input(
            "Nombre o identificador del potro",
            value="Potro evaluado",
            help="Nombre para identificar al animal en los resultados"
        )

        # st.radio para elegir sexo
        sexo_elegido = st.radio(
            "Sexo:",
            ["Macho ♂", "Hembra ♀"],
            horizontal=True
        )
        sexo_key = "M" if "Macho" in sexo_elegido else "H"

        st.markdown("**Pesos mensuales (kg):**")
        st.caption("Ingresa el peso de cada mes disponible. Deja en 0 si no hay medición.")

        # Diccionario para almacenar los pesos ingresados
        # Las llaves son la edad en meses y los valores son los pesos
        pesos_ingresados = {}

        # Crear un campo numérico para cada mes del 1 al 22
        # st.number_input crea un campo para ingresar números
        for mes in range(1, 23):
            valor = st.number_input(
                f"Mes {mes}",           # etiqueta visible
                min_value=0.0,         # valor mínimo permitido
                max_value=700.0,       # valor máximo (peso máximo plausible)
                value=0.0,             # valor inicial
                step=1.0,             # incremento al usar las flechas
                key=f"peso_mes_{mes}" # clave única para este widget
            )
            # Solo guardar si el usuario ingresó un valor mayor a 0
            if valor > 0:
                pesos_ingresados[mes] = valor

    # ── Columna derecha: resultados ───────────────────────────────────
    with col_resultado:

        # Si el usuario ingresó menos de 2 pesos, no hay suficiente
        # información para evaluar — mostrar mensaje informativo
        if len(pesos_ingresados) < 2:
            st.info(
                "Ingresa al menos **2 mediciones de peso** "
                "para ver la evaluación completa."
            )

        else:
            # Cargar estadísticas de referencia para el sexo elegido
            stats = stats_ref[f"stats_{sexo_key}"]

            # ── Calcular evaluación mes a mes ─────────────────────────
            # Para cada mes con dato, comparar contra los percentiles
            filas_evaluacion = []

            for edad in sorted(pesos_ingresados.keys()):
                peso = pesos_ingresados[edad]

                # Buscar los percentiles de referencia para esta edad
                # stats es un DataFrame, filtramos la fila del mes correcto
                referencia = stats[stats.edad_meses == int(edad)]

                # Si no hay referencia para esa edad, saltar
                if referencia.empty:
                    continue

                # Extraer valores de percentiles
                # .values[0] convierte el resultado de pandas a un número
                p10 = referencia["p10"].values[0]
                p25 = referencia["p25"].values[0]
                p50 = referencia["p50"].values[0]
                p75 = referencia["p75"].values[0]
                p90 = referencia["p90"].values[0]

                # Calcular diferencia porcentual respecto a la mediana
                # Fórmula: ((peso_real - mediana) / mediana) × 100
                # Un valor positivo = está por encima de la mediana
                # Un valor negativo = está por debajo de la mediana
                diferencia_pct = ((peso - p50) / p50) * 100

                # Clasificar en qué zona percentil está el peso
                if peso < p10:
                    zona = "MUY BAJO"
                    emoji = "🔴"
                    hay_alerta = True
                elif peso < p25:
                    zona = "BAJO"
                    emoji = "🟡"
                    hay_alerta = True
                elif peso <= p75:
                    zona = "NORMAL"
                    emoji = "🟢"
                    hay_alerta = False
                elif peso <= p90:
                    zona = "ALTO"
                    emoji = "🔵"
                    hay_alerta = False
                else:
                    zona = "MUY ALTO"
                    emoji = "🔵"
                    hay_alerta = True

                # Guardar todos los datos de esta medición
                filas_evaluacion.append({
                    "edad_meses": edad,
                    "peso_kg":    peso,
                    "p10":        round(p10, 1),
                    "P25":        round(p25, 1),
                    "P50":        round(p50, 1),
                    "P75":        round(p75, 1),
                    "p90":        round(p90, 1),
                    "vs_mediana": f"{diferencia_pct:+.1f}%",
                    "zona":       zona,
                    "emoji":      emoji,
                    "alerta":     hay_alerta,
                })

            # Convertir lista de diccionarios a DataFrame de pandas
            df_evaluacion = pd.DataFrame(filas_evaluacion)

            # ── Determinar patrón de crecimiento ─────────────────────
            # Calcular qué proporción de mediciones están en zona alta/baja
            proporcion_alto = df_evaluacion["zona"].str.contains("ALTO").mean()
            proporcion_bajo = df_evaluacion["zona"].str.contains("BAJO").mean()

            # Detectar irregularidad: pérdidas de peso consecutivas
            # Ordenar mediciones por edad para comparar consecutivos
            pesos_ordenados = [
                (edad, pesos_ingresados[edad])
                for edad in sorted(pesos_ingresados.keys())
            ]

            perdidas_consecutivas = 0
            hay_caida_brusca = False

            for i in range(1, len(pesos_ordenados)):
                edad_anterior, peso_anterior = pesos_ordenados[i-1]
                edad_actual,   peso_actual   = pesos_ordenados[i]

                # Solo comparar meses que estén cerca (máximo 3 meses de hueco)
                # Si hay un hueco de más de 3 meses, no comparamos
                # porque el animal pudo haber tenido un período normal
                # de recuperación que no capturamos
                if (edad_actual - edad_anterior) <= 3:

                    # Pérdida de peso: el actual pesa menos que el anterior
                    if peso_actual < peso_anterior:
                        perdidas_consecutivas += 1

                    # Caída brusca: pierde más del 8% en una sola medición
                    # Fórmula: cambio = (nuevo - anterior) / anterior × 100
                    cambio_porcentual = (
                        (peso_actual - peso_anterior) / peso_anterior * 100
                    )
                    if cambio_porcentual < -8:
                        hay_caida_brusca = True

            # Aplicar reglas de clasificación en orden de prioridad
            # Irregular tiene prioridad sobre todo
            es_irregular = (perdidas_consecutivas >= 4) or hay_caida_brusca

            if es_irregular:
                patron = "Irregular"
                color_patron = "#B71C1C"   # rojo oscuro
                explicacion = (
                    "El potro presentó pérdidas de peso en múltiples "
                    "mediciones consecutivas o una caída brusca mayor al 8%. "
                    "Esto es anormal en un potro en crecimiento y requiere "
                    "evaluación clínica de alimentación, sanidad y manejo."
                )
            elif proporcion_alto >= 0.60:
                patron = "Superior"
                color_patron = "#1565C0"   # azul
                explicacion = (
                    "El potro se mantuvo por encima del percentil 75 "
                    "en la mayoría de sus mediciones. "
                    "Crecimiento consistentemente superior al promedio del rancho."
                )
            elif proporcion_bajo >= 0.60:
                patron = "Inferior"
                color_patron = "#E65100"   # naranja oscuro
                explicacion = (
                    "El potro se mantuvo por debajo del percentil 25 "
                    "en la mayoría de sus mediciones. "
                    "Se recomienda revisar plan nutricional y estado sanitario."
                )
            else:
                patron = "Normal"
                color_patron = "#2E7D32"   # verde oscuro
                explicacion = (
                    "El potro se mantuvo dentro del rango normal (P25–P75) "
                    "en la mayoría de sus mediciones. "
                    "Crecimiento consistente con el promedio del rancho."
                )

            # ── Mostrar resultado principal ───────────────────────────
            st.subheader(f"Resultado: {nombre}")

            # Caja coloreada con el patrón detectado
            # unsafe_allow_html=True permite insertar HTML directamente
            st.markdown(
                f"<div style='"
                f"background-color:{color_patron};"
                f"padding:15px;"
                f"border-radius:10px;"
                f"color:white;"
                f"font-size:22px;"
                f"font-weight:bold;"
                f"text-align:center'>"
                f"Patrón de crecimiento: {patron}"
                f"</div>",
                unsafe_allow_html=True
            )

            # Espaciado
            st.markdown("")

            # Explicación clínica del patrón
            st.info(explicacion)

            # Mostrar alertas si las hay
            n_alertas = df_evaluacion["alerta"].sum()
            if n_alertas == 0:
                st.success(
                    f"Sin alertas — crecimiento dentro del rango "
                    f"esperado en todos los meses evaluados."
                )
            else:
                meses_con_alerta = df_evaluacion[
                    df_evaluacion["alerta"] == True
                ]["edad_meses"].tolist()
                st.warning(
                    f"⚠️ Alertas en **{n_alertas} meses**: {meses_con_alerta}"
                )

            # ── Gráfica del potro vs. curvas de referencia ────────────
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            color_ref = "#1565C0" if sexo_key == "M" else "#AD1457"

            # Zonas de referencia (mismo código que Sección 1)
            ax2.fill_between(
                stats.edad_meses, stats.p25, stats.p75,
                alpha=0.20, color=color_ref,
                label="Rango normal (P25–P75)"
            )
            ax2.fill_between(
                stats.edad_meses, stats.p10, stats.p90,
                alpha=0.08, color=color_ref,
                label="P10–P90"
            )

            # Línea de mediana de referencia
            ax2.plot(
                stats.edad_meses, stats.p50,
                color=color_ref, linewidth=2,
                linestyle="--", alpha=0.7,
                label="Mediana del rancho"
            )

            # Curva del potro evaluado
            # marker="o" dibuja un círculo en cada punto de medición
            ax2.plot(
                df_evaluacion["edad_meses"],
                df_evaluacion["peso_kg"],
                color=color_patron, linewidth=2.5,
                marker="o", markersize=7,
                label=f"{nombre} ({patron})",
                zorder=5   # zorder más alto = dibuja encima de las otras líneas
            )

            # Marcar los meses con alerta con una X roja
            alertas = df_evaluacion[df_evaluacion["alerta"] == True]
            if len(alertas) > 0:
                ax2.scatter(
                    alertas["edad_meses"],
                    alertas["peso_kg"],
                    color="red", s=120,
                    marker="x", linewidths=2.5,
                    zorder=6,
                    label="Alerta"
                )

            ax2.set_xlabel("Edad (meses)", fontsize=11)
            ax2.set_ylabel("Peso (kg)", fontsize=11)
            ax2.set_title(
                f"{nombre} — Curva real vs. Referencia del rancho",
                fontsize=12
            )
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 23)

            st.pyplot(fig2)
            plt.close(fig2)

            # ── Tabla de evaluación detallada ─────────────────────────
            st.markdown("---")
            st.subheader("Evaluación mes a mes")
            st.markdown(
                "La columna **vs Mediana** muestra cuánto se aparta "
                "el peso real del valor esperado para su edad y sexo. "
                "La columna **Estado** indica en qué zona percentil "
                "se encuentra cada medición."
            )

            # Preparar tabla para mostrar
            tabla_eval = df_evaluacion[[
                "edad_meses", "peso_kg", "P25", "P50", "P75",
                "vs_mediana", "zona", "alerta"
            ]].copy()

            tabla_eval.columns = [
                "Edad (meses)", "Peso real (kg)",
                "P25", "P50", "P75",
                "vs Mediana", "Estado", "Alerta"
            ]

            # Función para colorear las celdas de Estado
            # Esta función se aplica celda por celda
            def colorear_estado(valor):
                if "MUY BAJO" in str(valor) or "BAJO" in str(valor):
                    return "background-color: #FFCCBC"  # naranja claro
                elif "MUY ALTO" in str(valor):
                    return "background-color: #BBDEFB"  # azul claro
                elif "NORMAL" in str(valor):
                    return "background-color: #C8E6C9"  # verde claro
                return ""

            # Aplicar colores y mostrar tabla
            st.dataframe(
                tabla_eval.style.applymap(
                    colorear_estado, subset=["Estado"]
                ),
                use_container_width=True,
                hide_index=True
            )


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 3: PREDICTOR DE PESO
# ══════════════════════════════════════════════════════════════════

elif seccion == "🎯 Predictor de Peso":

    st.header("Predictor de Peso por Edad")
    st.markdown(
        "Basado en el modelo de regresión polinomial grado 3 "
        "entrenado con 3,981 mediciones reales. "
        "**R²=0.93** — el modelo explica el 93% de la variación de peso."
    )

    # ── Controles de entrada ──────────────────────────────────────────
    col_control, col_resultado_pred = st.columns(2)

    with col_control:
        st.subheader("Parámetros")

        sexo_pred = st.radio(
            "Sexo del potro:",
            ["Macho ♂", "Hembra ♀"],
            horizontal=True
        )
        sexo_bin = 1 if "Macho" in sexo_pred else 0
        sexo_key_pred = "M" if sexo_bin == 1 else "H"

        # st.slider crea un control deslizante
        # Parámetros: etiqueta, valor_min, valor_max, valor_inicial
        edad_pred = st.slider(
            "Edad a predecir (meses):",
            min_value=1,
            max_value=22,
            value=6,
            help="Mueve el deslizador para cambiar la edad"
        )

        st.markdown("---")
        st.markdown("**¿Cómo funciona el modelo?**")
        st.markdown(
            "El modelo recibe **sexo** (0=hembra, 1=macho) y "
            "**edad en meses** como variables de entrada. "
            "Internamente las transforma con un polinomio de grado 3 "
            "y aplica los coeficientes aprendidos durante el entrenamiento "
            "para calcular el peso esperado."
        )

    # ── Calcular predicción ───────────────────────────────────────────
    # modelo.predict espera una lista de listas: [[sexo_bin, edad]]
    # El resultado es un array de numpy, tomamos el primer elemento [0]
    peso_predicho = modelo.predict([[sexo_bin, edad_pred]])[0]

    # Obtener percentiles de referencia para comparar
    stats_pred = stats_ref[f"stats_{sexo_key_pred}"]
    ref_pred   = stats_pred[stats_pred.edad_meses == edad_pred]

    with col_resultado_pred:
        st.subheader("Resultado")

        # st.metric muestra un número grande con etiqueta
        st.metric(
            label=f"Peso predicho a los {edad_pred} meses",
            value=f"{peso_predicho:.0f} kg"
        )

        if len(ref_pred) > 0:
            p25_pred = ref_pred["p25"].values[0]
            p50_pred = ref_pred["p50"].values[0]
            p75_pred = ref_pred["p75"].values[0]
            n_pred   = ref_pred["n"].values[0]

            st.markdown(f"**Mediana real del rancho:** {p50_pred:.0f} kg")
            st.markdown(f"**Rango normal (P25–P75):** {p25_pred:.0f} – {p75_pred:.0f} kg")
            st.markdown(f"**Basado en:** {n_pred:.0f} animales reales")

            # Comparar predicción contra rango real
            diferencia = peso_predicho - p50_pred
            st.markdown(
                f"**Diferencia modelo vs. mediana real:** {diferencia:+.1f} kg"
            )

    # ── Gráfica de curvas de predicción ──────────────────────────────
    st.markdown("---")
    st.subheader("Curvas de predicción completas")

    # Generar predicciones para todas las edades de 1 a 22
    edades_completas = list(range(1, 23))

    # Lista de comprensión: genera una predicción por cada edad
    # para machos (sexo_bin=1) y hembras (sexo_bin=0)
    preds_machos  = [modelo.predict([[1, e]])[0] for e in edades_completas]
    preds_hembras = [modelo.predict([[0, e]])[0] for e in edades_completas]

    stats_M = stats_ref["stats_M"]
    stats_H = stats_ref["stats_H"]

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Zonas de referencia para machos
    ax3.fill_between(
        stats_M.edad_meses, stats_M.p25, stats_M.p75,
        alpha=0.12, color="#1565C0",
        label="Rango normal Machos (P25–P75)"
    )

    # Zonas de referencia para hembras
    ax3.fill_between(
        stats_H.edad_meses, stats_H.p25, stats_H.p75,
        alpha=0.12, color="#AD1457",
        label="Rango normal Hembras (P25–P75)"
    )

    # Curva de predicción machos
    ax3.plot(
        edades_completas, preds_machos,
        color="#1565C0", linewidth=2.5,
        label="Predicción Machos"
    )

    # Curva de predicción hembras
    ax3.plot(
        edades_completas, preds_hembras,
        color="#AD1457", linewidth=2.5,
        label="Predicción Hembras"
    )

    # Línea vertical en la edad seleccionada
    ax3.axvline(
        x=edad_pred, color="gray",
        linestyle="--", alpha=0.7,
        label=f"Mes {edad_pred} seleccionado"
    )

    # Punto de la predicción actual
    ax3.scatter(
        [edad_pred], [peso_predicho],
        color="orange", s=200, zorder=5,
        label=f"{peso_predicho:.0f} kg"
    )

    ax3.set_xlabel("Edad (meses)", fontsize=12)
    ax3.set_ylabel("Peso (kg)", fontsize=12)
    ax3.set_title(
        "Curvas de predicción del modelo vs. Rangos de referencia real\n"
        "Las zonas sombreadas son los rangos normales observados en el rancho",
        fontsize=11
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, 22)
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Tabla completa de predicciones ───────────────────────────────
    st.markdown("---")
    st.subheader("Tabla de predicciones para todas las edades")

    tabla_pred = pd.DataFrame({
        "Edad (meses)":           edades_completas,
        "Prediccion Machos (kg)": [round(p) for p in preds_machos],
        "Prediccion Hembras (kg)":[round(p) for p in preds_hembras],
    })

    st.dataframe(tabla_pred, use_container_width=True, hide_index=True)

    # ── Métricas del modelo ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Métricas de validación del modelo")
    st.markdown(
        "Estas métricas demuestran la confiabilidad del modelo. "
        "Se calcularon sobre el **20% de datos de prueba** que el modelo "
        "nunca vio durante el entrenamiento."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("R² (coef. determinación)", "0.9324",
                  help="Proporción de variación explicada. 1.0 = perfecto.")
    with col_m2:
        st.metric("MAE (error promedio)", "19.6 kg",
                  help="Error absoluto medio en las predicciones.")
    with col_m3:
        st.metric("Datos de entrenamiento", "3,181",
                  help="Mediciones usadas para entrenar el modelo.")