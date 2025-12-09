"""
DemenciaScan - Aplicación Web para Detección de Demencia con IA
Interfaz gráfica moderna con Streamlit
"""

import streamlit as st
import os
import sys
import tempfile
import wave
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json

# Agregar directorio actual al path
sys.path.append(str(Path(__file__).parent))

# Importar módulos locales
from config import (
    STREAMLIT_CONFIG,
    MESSAGES,
    MODEL_PATH,
    GROQ_API_KEY,
    DEMOGRAPHIC_FEATURES,
    CLASSIFICATION_THRESHOLDS,
    THEME_COLORS
)

from utils_gui import (
    create_probability_gauge,
    create_comparison_bars,
    create_indicators_radar,
    create_timeline_chart,
    format_prediction_result,
    format_indicators_table,
    get_interpretation_message,
    validate_text_input,
    validate_demographic_data,
    export_to_json,
    export_to_csv,
    generate_pdf_report,
    save_to_history,
    load_history,
    get_example_transcription
)

# Importar módulos de DemenciaScan
sys.path.append(str(Path(__file__).parent / "DemenciaScan"))
from DemenciaScan.predict_new_transcription import DemenciaPredictor

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

# ============================================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #534;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN DE ESTADO
# ============================================================================

if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

if 'current_transcription' not in st.session_state:
    st.session_state.current_transcription = ""

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'history' not in st.session_state:
    st.session_state.history = load_history()

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def load_model():
    """Carga el modelo BERT (con caché)"""
    try:
        if not MODEL_PATH.exists():
            return None, "Modelo no encontrado"
        
        predictor = DemenciaPredictor(str(MODEL_PATH))
        return predictor, "Modelo cargado correctamente"
    except Exception as e:
        return None, f"Error al cargar modelo: {str(e)}"


def check_system_status():
    """Verifica el estado del sistema"""
    status = {
        "model": False,
        "api": False,
        "messages": []
    }
    
    # Verificar modelo
    if MODEL_PATH.exists():
        status["model"] = True
        status["messages"].append("✅ Modelo BERT disponible")
    else:
        status["messages"].append("❌ Modelo BERT no encontrado")
    
    # Verificar API
    if GROQ_API_KEY and GROQ_API_KEY != "":
        status["api"] = True
        status["messages"].append("✅ API de Groq configurada")
    else:
        status["messages"].append("⚠️ API de Groq no configurada")
    
    return status


def process_text_analysis(text: str, demographics: dict):
    """Procesa análisis de texto"""
    try:
        # Validar entrada
        is_valid, message = validate_text_input(text)
        if not is_valid:
            return None, message
        
        is_valid, message = validate_demographic_data(demographics)
        if not is_valid:
            return None, message
        
        # Cargar modelo si no está cargado
        if st.session_state.predictor is None:
            predictor, msg = load_model()
            if predictor is None:
                return None, msg
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            temp_file = f.name
        
        # Hacer predicción
        results = st.session_state.predictor.predict_single_file(temp_file, demographics)
        
        # Limpiar archivo temporal
        os.unlink(temp_file)
        
        return results, "Análisis completado"
        
    except Exception as e:
        return None, f"Error en el análisis: {str(e)}"


def render_results_dashboard(results):
    """Renderiza el dashboard de resultados"""
    # Resultado principal
    st.markdown(format_prediction_result(results), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizaciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_probability_gauge(
                results['probabilidad_con_riesgo'],
                "Probabilidad de Riesgo"
            ),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_comparison_bars(
                results['probabilidad_sin_riesgo'],
                results['probabilidad_con_riesgo']
            ),
            use_container_width=True
        )
    
    # Indicadores clave
    st.markdown("### 🔍 Indicadores Clave Detectados")
    
    indicators_df = format_indicators_table(results['indicadores_clave'])
    st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    
    # Gráfico de radar
    st.plotly_chart(
        create_indicators_radar(results['indicadores_clave']),
        use_container_width=True
    )
    
    # Interpretación
    st.markdown("### 💡 Interpretación")
    interpretation = get_interpretation_message(results['probabilidad_con_riesgo'])
    
    if results['probabilidad_con_riesgo'] >= CLASSIFICATION_THRESHOLDS["high_risk"]:
        st.markdown(f'<div class="danger-box">{interpretation}</div>', unsafe_allow_html=True)
    elif results['probabilidad_con_riesgo'] >= CLASSIFICATION_THRESHOLDS["moderate_risk"]:
        st.markdown(f'<div class="warning-box">{interpretation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box">{interpretation}</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="info-box">{MESSAGES["disclaimer"]}</div>', unsafe_allow_html=True)
    
    # Opciones de exportación
    st.markdown("---")
    st.markdown("### 💾 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Exportar JSON", key=f"json_{results.get('fecha', 'current')}"):
            filepath = export_to_json(results)
            st.success(f"✅ Exportado a: {filepath}")
    
    with col2:
        if st.button("📊 Exportar CSV", key=f"csv_{results.get('fecha', 'current')}"):
            filepath = export_to_csv(results)
            st.success(f"✅ Exportado a: {filepath}")
    
    with col3:
        if st.button("📑 Generar PDF", key=f"pdf_{results.get('fecha', 'current')}"):
            try:
                filepath = generate_pdf_report(results)
                st.success(f"✅ PDF generado: {filepath}")
            except Exception as e:
                st.error(f"❌ Error al generar PDF: {str(e)}")


def render_demographics_form(key_prefix=""):
    """Renderiza el formulario de datos demográficos"""
    st.markdown("### 👤 Información Demográfica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.number_input(
            "Edad del paciente:",
            min_value=0,
            max_value=120,
            value=70,
            step=1,
            key=f"{key_prefix}_edad"
        )
        
        nivel_educacion = st.selectbox(
            "Nivel de educación:",
            options=DEMOGRAPHIC_FEATURES["nivel_educacion"]["options"],
            index=1,
            key=f"{key_prefix}_educacion"
        )
        
        calidad_sueno = st.selectbox(
            "Calidad de sueño:",
            options=DEMOGRAPHIC_FEATURES["calidad_sueno"]["options"],
            index=1,
            key=f"{key_prefix}_sueno"
        )
    
    with col2:
        habitos_diarios = st.selectbox(
            "Hábitos diarios:",
            options=DEMOGRAPHIC_FEATURES["habitos_diarios"]["options"],
            index=0,
            key=f"{key_prefix}_habitos"
        )
        
        nivel_entorno_social = st.selectbox(
            "Nivel de entorno social:",
            options=DEMOGRAPHIC_FEATURES["nivel_entorno_social"]["options"],
            index=1,
            key=f"{key_prefix}_social"
        )
    
    return {
        'edad': edad,
        'nivel_educacion': nivel_educacion,
        'calidad_sueno': calidad_sueno,
        'habitos_diarios': habitos_diarios,
        'nivel_entorno_social': nivel_entorno_social
    }


# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================

st.sidebar.markdown("# 🧠 DemenciaScan")
st.sidebar.markdown("---")

# Menú de navegación
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "📝 Análisis de Texto", "🎤 Transcripción", "🧠 Análisis Completo", 
     "📊 Historial", "⚙️ Configuración", "ℹ️ Ayuda"]
)

st.sidebar.markdown("---")

# Estado del sistema
st.sidebar.markdown("### 📊 Estado del Sistema")
system_status = check_system_status()

for msg in system_status["messages"]:
    st.sidebar.markdown(msg)

st.sidebar.markdown("---")

# Información adicional
st.sidebar.markdown("### 📚 Recursos")
st.sidebar.markdown("- [Documentación](README.md)")
st.sidebar.markdown("- [Guía de Uso](GUIA_EJECUCION.md)")
st.sidebar.markdown("- [GitHub](https://github.com)")

# ============================================================================
# PÁGINA: INICIO
# ============================================================================

if page == "🏠 Inicio":
    st.markdown('<h1 class="main-header">🧠 DemenciaScan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de Detección de Demencia con Inteligencia Artificial</p>', unsafe_allow_html=True)
    
    # Descripción
    st.markdown("""
    <div class="info-box" style="color: #333; background-color: #e0f7fa; border-left: 5px solid #00796b; padding: 1rem; border-radius: 8px;">
        <h3>¿Qué es DemenciaScan?</h3>
        <p>
            DemenciaScan es una herramienta de apoyo clínico que utiliza modelos de lenguaje BERT 
            para analizar transcripciones de conversaciones y detectar posibles signos de deterioro cognitivo 
            asociados con demencia.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Características principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎤 Transcripción
        - Grabación de audio
        - Transcripción automática
        - Detección de pausas
        - Análisis de patrones
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Análisis IA
        - Modelo BERT multilingüe
        - Características lingüísticas
        - Datos demográficos
        - Predicción de riesgo
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Resultados
        - Probabilidades detalladas
        - Indicadores clave
        - Visualizaciones
        - Exportación de reportes
        """)
    
    st.markdown("---")
    
    # Cómo empezar
    st.markdown("### 🚀 Cómo Empezar")
    
    st.markdown("""
    1. **Análisis de Texto**: Si ya tienes una transcripción, ve a la sección de Análisis de Texto
    2. **Transcripción**: Para grabar y transcribir audio, usa la sección de Transcripción
    3. **Análisis Completo**: Para un flujo integrado de transcripción + análisis
    4. **Historial**: Revisa análisis previos y compara resultados
    """)
    
    # Advertencia importante
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Advertencia Importante</h4>
        <p>
        Esta herramienta es para fines de apoyo clínico e investigación. 
        <strong>NO reemplaza el diagnóstico médico profesional</strong>. 
        Los resultados deben ser interpretados por profesionales de la salud calificados.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estadísticas rápidas
    st.markdown("---")
    st.markdown("### 📈 Estadísticas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Análisis Realizados", len(st.session_state.history))
    
    with col2:
        if st.session_state.model_loaded:
            st.metric("Estado del Modelo", "✅ Cargado")
        else:
            st.metric("Estado del Modelo", "⏳ No cargado")
    
    with col3:
        st.metric("Precisión del Modelo", "~85%")
    
    with col4:
        st.metric("Idiomas Soportados", "ES, EN, Multi")

# ============================================================================
# PÁGINA: ANÁLISIS DE TEXTO
# ============================================================================

elif page == "📝 Análisis de Texto":
    st.markdown("# 📝 Análisis de Texto")
    st.markdown("Analiza una transcripción existente para detectar posibles signos de demencia")
    
    st.markdown("---")
    
    # Opciones de entrada
    input_method = st.radio(
        "Método de entrada:",
        ["📄 Cargar archivo .txt", "✍️ Pegar texto directamente", "📋 Usar ejemplo"]
    )
    
    text_to_analyze = ""
    
    if input_method == "📄 Cargar archivo .txt":
        uploaded_file = st.file_uploader("Selecciona un archivo de texto", type=['txt'])
        if uploaded_file is not None:
            text_to_analyze = uploaded_file.read().decode('utf-8')
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            with st.expander("Ver contenido"):
                st.text_area("Contenido del archivo:", text_to_analyze, height=200, disabled=True)
    
    elif input_method == "✍️ Pegar texto directamente":
        text_to_analyze = st.text_area(
            "Pega aquí la transcripción:",
            height=300,
            placeholder="Ejemplo: Hola, estoy describiendo lo que veo en esta imagen..."
        )
    
    else:  # Usar ejemplo
        st.info("📋 Usando transcripción de ejemplo")
        text_to_analyze = get_example_transcription()
        with st.expander("Ver transcripción de ejemplo"):
            st.text_area("Ejemplo:", text_to_analyze, height=200, disabled=True)
    
    st.markdown("---")
    
    # Formulario demográfico
    demographics = render_demographics_form("text_analysis")
    
    st.markdown("---")
    
    # Botón de análisis
    if st.button("🔍 Analizar Texto", type="primary"):
        if not text_to_analyze or len(text_to_analyze.strip()) == 0:
            st.error("❌ Por favor, proporciona un texto para analizar")
        else:
            with st.spinner("🔍 Analizando transcripción..."):
                results, message = process_text_analysis(text_to_analyze, demographics)
                
                if results is None:
                    st.error(f"❌ {message}")
                else:
                    st.session_state.analysis_results = results
                    st.success("✅ Análisis completado")
                    
                    # Guardar en historial
                    save_to_history(results)
                    st.session_state.history = load_history()
    
    # Mostrar resultados si existen
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.markdown("## 📊 Resultados del Análisis")
        
        render_results_dashboard(st.session_state.analysis_results)

# ============================================================================
# PÁGINA: TRANSCRIPCIÓN
# ============================================================================

elif page == "🎤 Transcripción":
    st.markdown("# 🎤 Transcripción de Audio")
    st.markdown("Graba y transcribe audio para análisis posterior")

    st.markdown("---")

    # Importar funciones de dictar.py
    import tempfile
    import wave
    import pyaudio
    import keyboard
    import pyperclip
    from groq import Groq

    # Configurar cliente Groq
    groq_client = Groq(api_key=GROQ_API_KEY)

    # Funciones de grabación y transcripción (adaptadas de dictar.py)
    def grabar_audio(frecuencia_muestreo=16000, canales=1, fragmento=1024):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=canales,
            rate=frecuencia_muestreo,
            input=True,
            frames_per_buffer=fragmento,)
        print("Presiona o manten presionado el boton INS para comenzar a grabar")
        frames = []
        keyboard.wait("insert")
        print("Grabando... (suelta INS para deterner)")
        while keyboard.is_pressed("insert"):
            data = stream.read(fragmento)
            frames.append(data)
        print("Garbación finalizada")
        stream.stop_stream()
        stream.close()
        p.terminate()
        return frames, frecuencia_muestreo

    def guardar_audio(frames, frecuencia_muestreo):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
            wf = wave.open(audio_temp.name, mode="wb")
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(frecuencia_muestreo)
            wf.writeframes(b"".join(frames))
            wf.close()
            return audio_temp.name

    def transcribir_audio(ruta_archivo_audio):
        try:
            with open(ruta_archivo_audio, "rb") as archivo:
                transcripcion = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(ruta_archivo_audio), archivo.read()),
                    model="whisper-large-v3",
                    prompt= "Transcribe el audio de una conversación en español con alta precisión, prestando especial atención a capturar pausas, repeticiones, autocorrecciones, errores de pronunciación y cualquier anomalía en el habla que pueda ser relevante para un análisis clínico de posibles signos de demencia o deterioro cognitivo. Marca claramente las pausas, incluyendo incluso las pausas muy breves o mínimas (por ejemplo, desde 0.2 segundos de silencio o menos), usando símbolos como '...' para pausas prolongadas y '[pausa breve]' para pausas cortas. También indica repeticiones (por ejemplo, palabras repetidas) y reformulaciones. El objetivo es obtener una transcripción detallada y fiel que permita un análisis lingüístico y discursivo posterior.",
                    response_format="text",
                    language="es"
                )
            return transcripcion
        except Exception as e:
            print(f"Ocurrio un error: {str(e)}")
            return None

    # Opciones de grabación
    st.markdown("### 🎙️ Opciones de Grabación")

    recording_method = st.radio(
        "Método de grabación:",
        ["🎤 Grabación desde micrófono (teclado)", "🎙️ Grabación en navegador (experimental)"]
    )

    if recording_method == "🎤 Grabación desde micrófono (teclado)":
        st.info("""
        **Instrucciones:**
        1. Haz clic en "Iniciar Grabación"
        2. Presiona y mantén la tecla **INS** para grabar
        3. Suelta **INS** para detener la grabación
        4. El audio se transcribirá automáticamente
        """)

        if st.button("🎤 Iniciar Grabación", type="primary"):
            with st.spinner("🎙️ Esperando señal de grabación..."):
                try:
                    frames, frecuencia_mestreo = grabar_audio()
                    if frames:
                        archivo_audio_temp = guardar_audio(frames, frecuencia_mestreo)

                        with st.spinner("🔍 Transcribiendo audio..."):
                            transcripcion = transcribir_audio(archivo_audio_temp)

                        if transcripcion:
                            st.session_state.current_transcription = transcripcion
                            st.success("✅ Transcripción completada")

                            # Copiar al portapapeles
                            pyperclip.copy(transcripcion)
                            st.info("📋 Transcripción copiada al portapapeles")

                            # Mostrar transcripción
                            st.markdown("### 📝 Transcripción Generada")
                            st.text_area(
                                "Transcripción:",
                                transcripcion,
                                height=200,
                                disabled=True
                            )
                        else:
                            st.error("❌ Error en la transcripción")

                        # Limpiar archivo temporal
                        os.unlink(archivo_audio_temp)
                    else:
                        st.warning("⚠️ No se grabó audio")

                except Exception as e:
                    st.error(f"❌ Error durante la grabación: {str(e)}")

    else:  # Grabación en navegador
        st.warning("⚠️ Esta funcionalidad está en desarrollo y puede no funcionar correctamente.")

        try:
            from streamlit_webrtc import webrtc_streamer
            import av

            def audio_frame_callback(frame):
                sound = pydub.AudioSegment.empty()
                sound += pydub.AudioSegment(
                    data=frame.to_ndarray().tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
                return av.AudioFrame.from_ndarray(
                    np.array(sound.get_array_of_samples()).reshape(1, -1),
                    layout=frame.layout.name
                )

            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_frame_callback=audio_frame_callback,
                media_stream_constraints={"audio": True, "video": False},
            )

            if webrtc_ctx.audio_receiver:
                st.write("🎙️ Grabando audio...")

                # Aquí se procesaría el audio recibido
                # Por ahora, solo mostramos un placeholder
                st.info("🎵 Audio recibido. Procesamiento en desarrollo...")

        except ImportError:
            st.error("❌ streamlit-webrtc no está instalado. Instala con: pip install streamlit-webrtc")

    st.markdown("---")

    # Área para pegar transcripción manual
    st.markdown("### 📋 Pegar Transcripción Manual")

    transcription = st.text_area(
        "O pega aquí una transcripción existente:",
        value=st.session_state.current_transcription if 'current_transcription' in st.session_state else "",
        height=200,
        placeholder="Pega aquí la transcripción..."
    )

    if transcription and transcription != st.session_state.get('current_transcription', ''):
        st.session_state.current_transcription = transcription
        st.success("✅ Transcripción guardada")

    # Navegación a análisis
    if st.session_state.get('current_transcription'):
        st.markdown("---")
        st.success("✅ Transcripción lista para análisis")

        st.markdown("### 📋 Próximos Pasos")
        st.markdown("""
        **Opción 1**: Ve a la sección "📝 Análisis de Texto" en el menú lateral para analizar esta transcripción

        **Opción 2**: Ve a la sección "🧠 Análisis Completo" para un flujo integrado
        """)

        # Información adicional
        with st.expander("ℹ️ Información sobre la transcripción"):
            st.markdown(f"**Longitud**: {len(st.session_state.current_transcription)} caracteres")
            st.markdown(f"**Palabras**: {len(st.session_state.current_transcription.split())} palabras")
            st.markdown("**Estado**: Lista para análisis")
            
        st.markdown("---")
        st.markdown("### 🔍 Análisis Inmediato")
        st.info("Completa los datos demográficos para analizar esta transcripción ahora mismo.")
        
        # Formulario demográfico para transcripción
        transcription_demographics = render_demographics_form("transcription")
        
        if st.button("🚀 Analizar Transcripción Ahora", type="primary"):
            with st.spinner("🔍 Analizando transcripción..."):
                results, message = process_text_analysis(st.session_state.current_transcription, transcription_demographics)
                
                if results is None:
                    st.error(f"❌ {message}")
                else:
                    st.session_state.analysis_results = results
                    st.success("✅ Análisis completado")
                    
                    # Guardar en historial
                    save_to_history(results)
                    st.session_state.history = load_history()
                    
                    # Mostrar resultados
                    st.markdown("---")
                    st.markdown("## 📊 Resultados del Análisis")
                    render_results_dashboard(results)

# ============================================================================
# PÁGINA: ANÁLISIS COMPLETO
# ============================================================================

elif page == "🧠 Análisis Completo":
    st.markdown("# 🧠 Análisis Completo")
    st.markdown("Flujo integrado: Transcripción + Análisis")
    
    st.markdown("---")
    
    st.info("""
    Esta sección integrará la grabación de audio, transcripción y análisis en un solo flujo.
    Por ahora, usa las secciones individuales de Transcripción y Análisis de Texto.
    """)
    
    # Mostrar flujo de trabajo
    st.markdown("### 📋 Flujo de Trabajo Recomendado")
    
    st.markdown("""
    1. **🎤 Transcripción**: Graba el audio usando `dictar.py`
    2. **📝 Análisis de Texto**: Pega la transcripción y completa datos demográficos
    3. **📊 Resultados**: Revisa las predicciones y visualizaciones
    4. **💾 Exportar**: Guarda los resultados en el formato deseado
    5. **📈 Historial**: Compara con análisis previos
    """)

# ============================================================================
# PÁGINA: HISTORIAL
# ============================================================================

elif page == "📊 Historial":
    st.markdown("# 📊 Historial de Análisis")
    st.markdown("Revisa y compara análisis previos")
    
    st.markdown("---")
    
    if len(st.session_state.history) == 0:
        st.info("📭 No hay análisis previos en el historial")
    else:
        st.success(f"📚 {len(st.session_state.history)} análisis en el historial")
        
        # Gráfico de evolución temporal
        if len(st.session_state.history) > 1:
            st.markdown("### 📈 Evolución Temporal")
            st.plotly_chart(
                create_timeline_chart(st.session_state.history),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Tabla de historial
        st.markdown("### 📋 Lista de Análisis")
        
        # Convertir a DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        
        # Seleccionar columnas relevantes
        display_columns = ['fecha', 'prediccion', 'probabilidad_con_riesgo', 'confianza']
        available_columns = [col for col in display_columns if col in history_df.columns]
        
        if available_columns:
            display_df = history_df[available_columns].copy()
            
            # Formatear
            if 'fecha' in display_df.columns:
                display_df['fecha'] = pd.to_datetime(display_df['fecha']).dt.strftime('%Y-%m-%d %H:%M')
            if 'probabilidad_con_riesgo' in display_df.columns:
                display_df['probabilidad_con_riesgo'] = (display_df['probabilidad_con_riesgo'] * 100).round(1).astype(str) + '%'
            if 'confianza' in display_df.columns:
                display_df['confianza'] = (display_df['confianza'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🔍 Ver Detalles de Análisis")
        
        # Selector de análisis
        analysis_options = {
            f"{h['fecha']} - {h.get('prediccion', 'N/A')} ({h.get('probabilidad_con_riesgo', 0)*100:.1f}%)": i 
            for i, h in enumerate(st.session_state.history)
        }
        
        selected_option = st.selectbox(
            "Selecciona un análisis para ver detalles:",
            options=list(analysis_options.keys()),
            index=None,
            placeholder="Selecciona un análisis..."
        )
        
        if selected_option is not None:
            selected_index = analysis_options[selected_option]
            selected_analysis = st.session_state.history[selected_index]
            
            st.markdown("---")
            st.markdown(f"### 📊 Detalles del Análisis: {selected_option}")
            render_results_dashboard(selected_analysis)
        
        # Botón para limpiar historial
        st.markdown("---")
        if st.button("🗑️ Limpiar Historial", type="secondary"):
            from utils_gui import clear_history
            clear_history()
            st.session_state.history = []
            st.success("✅ Historial limpiado")
            st.rerun()

# ============================================================================
# PÁGINA: CONFIGURACIÓN
# ============================================================================

elif page == "⚙️ Configuración":
    st.markdown("# ⚙️ Configuración")
    st.markdown("Ajusta los parámetros del sistema")
    
    st.markdown("---")
    
    # Configuración del modelo
    st.markdown("### 🤖 Modelo BERT")
    
    model_type = st.selectbox(
        "Tipo de modelo:",
        options=["multilingual", "spanish", "english"],
        index=0
    )
    
    st.info(f"Modelo actual: {model_type}")
    
    # Configuración de API
    st.markdown("---")
    st.markdown("### 🔑 API de Groq")
    
    api_key = st.text_input(
        "API Key:",
        value=GROQ_API_KEY if GROQ_API_KEY else "",
        type="password"
    )
    
    if st.button("💾 Guardar API Key"):
        # Aquí se guardaría en .env
        st.success("✅ API Key guardada (funcionalidad en desarrollo)")
    
    # Umbrales de clasificación
    st.markdown("---")
    st.markdown("### 🎯 Umbrales de Clasificación")
    
    high_risk_threshold = st.slider(
        "Umbral de Alto Riesgo:",
        min_value=0.0,
        max_value=1.0,
        value=CLASSIFICATION_THRESHOLDS["high_risk"],
        step=0.05
    )
    
    moderate_risk_threshold = st.slider(
        "Umbral de Riesgo Moderado:",
        min_value=0.0,
        max_value=1.0,
        value=CLASSIFICATION_THRESHOLDS["moderate_risk"],
        step=0.05
    )
    
    # Información del sistema
    st.markdown("---")
    st.markdown("### 💻 Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Modelo Cargado", "✅ Sí" if st.session_state.model_loaded else "❌ No")
        st.metric("API Configurada", "✅ Sí" if GROQ_API_KEY else "❌ No")
    
    with col2:
        st.metric("Análisis en Historial", len(st.session_state.history))
        st.metric("Versión", "1.0.0")

# ============================================================================
# PÁGINA: AYUDA
# ============================================================================

elif page == "ℹ️ Ayuda":
    st.markdown("# ℹ️ Ayuda y Documentación")
    
    st.markdown("---")
    
    # FAQ
    st.markdown("### ❓ Preguntas Frecuentes")
    
    with st.expander("¿Cómo funciona el sistema?"):
        st.markdown("""
        DemenciaScan utiliza un modelo BERT multilingüe entrenado para analizar 
        transcripciones de conversaciones y detectar patrones lingüísticos asociados 
        con deterioro cognitivo. El sistema analiza:
        
        - Características lingüísticas (riqueza léxica, repeticiones, pausas)
        - Indicadores de demencia (errores semánticos, problemas de comprensión)
        - Datos demográficos (edad, educación, hábitos)
        """)
    
    with st.expander("¿Qué tan preciso es el modelo?"):
        st.markdown("""
        El modelo tiene una precisión aproximada del 85% en datasets balanceados.
        Sin embargo, es importante recordar que:
        
        - Es una herramienta de apoyo, no un diagnóstico definitivo
        - Debe ser interpretado por profesionales médicos
        - La precisión puede variar según la calidad del audio y la transcripción
        """)
    
    with st.expander("¿Cómo interpretar los resultados?"):
        st.markdown("""
        Los resultados incluyen:
        
        - **Probabilidad de Riesgo**: Porcentaje de probabilidad de deterioro cognitivo
        - **Indicadores Clave**: Características específicas detectadas
        - **Nivel de Riesgo**: Alto (>70%), Moderado (50-70%), Bajo (<50%)
        
        Siempre consulta con un profesional médico para interpretación definitiva.
        """)
    
    with st.expander("¿Cómo grabar audio?"):
        st.markdown("""
        Actualmente, usa el script `dictar.py`:
        
        ```bash
        python dictar.py
        ```
        
        1. Presiona y mantén INS para grabar
        2. Suelta INS para detener
        3. La transcripción se copia automáticamente
        """)
    
    # Contacto y soporte
    st.markdown("---")
    st.markdown("### 📞 Soporte")
    
    st.info("""
    Para soporte técnico o preguntas:
    - Revisa la documentación en `GUIA_EJECUCION.md`
    - Consulta el README del proyecto
    - Contacta al equipo de desarrollo
    """)
    
    # Enlaces útiles
    st.markdown("---")
    st.markdown("### 🔗 Enlaces Útiles")
    
    st.markdown("""
    - [Documentación Completa](README.md)
    - [Guía de Ejecución](GUIA_EJECUCION.md)
    - [Groq API Docs](https://console.groq.com/docs)
    - [Streamlit Docs](https://docs.streamlit.io/)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>DemenciaScan v1.0.0 | Desarrollado con ❤️ para la detección temprana de demencia</p>
    <p style="font-size: 0.8rem;">⚠️ Herramienta de apoyo clínico - No reemplaza diagnóstico médico profesional</p>
</div>
""", unsafe_allow_html=True)
