"""
Configuración centralizada para el proyecto de Detección de Demencia
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Directorios principales
DEMENCIA_SCAN_DIR = ROOT_DIR / "DemenciaScan"
MODELS_DIR = DEMENCIA_SCAN_DIR / "models_test"
DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = ROOT_DIR / "temp"
EXPORTS_DIR = ROOT_DIR / "exports"

# Crear directorios si no existen
for directory in [DATA_DIR, TEMP_DIR, EXPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Archivos importantes
MODEL_PATH = MODELS_DIR / "best_model.pth"
TRAINING_DATA_CSV = ROOT_DIR / "transcripciones_procesadas.csv"

# ============================================================================
# CONFIGURACIÓN DE API
# ============================================================================

# Groq API (para transcripción)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

# Modelos BERT disponibles
BERT_MODELS = {
    "multilingual": "bert-base-multilingual-cased",
    "spanish": "dccuchile/bert-base-spanish-wwm-cased",
    "english": "bert-base-uncased"
}

# Modelo por defecto
DEFAULT_BERT_MODEL = "multilingual"

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "num_epochs": 20,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "max_length": 256,
    "warmup_steps": 100,
    "weight_decay": 0.01
}

# ============================================================================
# CONFIGURACIÓN DE AUDIO
# ============================================================================

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "paInt16",
    "max_duration": 300  # 5 minutos máximo
}

# Configuración de Whisper (Groq)
WHISPER_CONFIG = {
    "model": "whisper-large-v3",
    "language": "es",
    "response_format": "text",
    "prompt": (
        "Transcribe el audio de una conversación en español con alta precisión, "
        "prestando especial atención a capturar pausas, repeticiones, autocorrecciones, "
        "errores de pronunciación y cualquier anomalía en el habla que pueda ser relevante "
        "para un análisis clínico de posibles signos de demencia o deterioro cognitivo. "
        "Marca claramente las pausas, incluyendo incluso las pausas muy breves o mínimas "
        "(por ejemplo, desde 0.2 segundos de silencio o menos), usando símbolos como '...' "
        "para pausas prolongadas y '[pausa breve]' para pausas cortas. También indica "
        "repeticiones (por ejemplo, palabras repetidas) y reformulaciones. El objetivo es "
        "obtener una transcripción detallada y fiel que permita un análisis lingüístico "
        "y discursivo posterior."
    )
}

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS
# ============================================================================

# Umbrales de clasificación
CLASSIFICATION_THRESHOLDS = {
    "high_risk": 0.7,      # > 70% = Alto riesgo
    "moderate_risk": 0.5,  # 50-70% = Riesgo moderado
    "low_risk": 0.5        # < 50% = Bajo riesgo
}

# Características demográficas
DEMOGRAPHIC_FEATURES = {
    "edad": {
        "type": "numeric",
        "min": 0,
        "max": 120,
        "default": 70
    },
    "nivel_educacion": {
        "type": "categorical",
        "options": ["primaria", "secundaria", "universitaria"],
        "mapping": {"primaria": 1, "secundaria": 2, "universitaria": 3},
        "default": "secundaria"
    },
    "calidad_sueno": {
        "type": "categorical",
        "options": ["mala", "regular", "buena"],
        "mapping": {"mala": 1, "regular": 2, "buena": 3},
        "default": "regular"
    },
    "habitos_diarios": {
        "type": "categorical",
        "options": ["sedentario", "ejercicio regular"],
        "mapping": {"sedentario": 1, "ejercicio regular": 2},
        "default": "sedentario"
    },
    "nivel_entorno_social": {
        "type": "categorical",
        "options": ["bajo", "medio", "alto"],
        "mapping": {"bajo": 1, "medio": 2, "alto": 3},
        "default": "medio"
    }
}

# Características lingüísticas a extraer
LINGUISTIC_FEATURES = [
    "num_palabras",
    "riqueza_lexica",
    "num_sentences",
    "avg_sentence_length",
    "num_questions",
    "num_exclamations",
    "repetition_ratio"
]

# Indicadores de demencia
DEMENTIA_INDICATORS = [
    "audio_comprehension_issues",
    "repetitive_patterns",
    "incomplete_thoughts",
    "filler_words_count",
    "language_switching",
    "semantic_errors",
    "phonemic_errors",
    "word_finding_pauses"
]

# Métricas del discurso
DISCOURSE_METRICS = [
    "total_utterances",
    "avg_utterance_length",
    "coherence_breaks",
    "topic_maintenance"
]

# Todas las características adicionales para el modelo
ADDITIONAL_FEATURES = (
    list(DEMOGRAPHIC_FEATURES.keys()) +
    LINGUISTIC_FEATURES
)

# Normalización de características (valores aproximados del entrenamiento)
FEATURE_NORMALIZATION = {
    "means": {
        "edad": 75,
        "nivel_educacion": 2,
        "calidad_sueno": 2,
        "habitos_diarios": 1.4,
        "nivel_entorno_social": 2,
        "num_palabras": 50,
        "riqueza_lexica": 0.6,
        "num_sentences": 5,
        "avg_sentence_length": 8,
        "num_questions": 1,
        "num_exclamations": 0.5,
        "repetition_ratio": 0.1
    },
    "stds": {
        "edad": 10,
        "nivel_educacion": 0.8,
        "calidad_sueno": 0.8,
        "habitos_diarios": 0.5,
        "nivel_entorno_social": 0.8,
        "num_palabras": 30,
        "riqueza_lexica": 0.2,
        "num_sentences": 3,
        "avg_sentence_length": 5,
        "num_questions": 1,
        "num_exclamations": 0.5,
        "repetition_ratio": 0.1
    }
}

# ============================================================================
# CONFIGURACIÓN DE INTERFAZ GRÁFICA
# ============================================================================

# Streamlit
STREAMLIT_CONFIG = {
    "page_title": "DemenciaScan - Detección de Demencia con IA",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Colores y tema
THEME_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ff9800",
    "danger": "#d62728",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

# Configuración de gráficos
PLOT_CONFIG = {
    "template": "plotly_white",
    "height": 400,
    "margin": {"l": 50, "r": 50, "t": 50, "b": 50}
}

# ============================================================================
# CONFIGURACIÓN DE EXPORTACIÓN
# ============================================================================

EXPORT_CONFIG = {
    "pdf": {
        "enabled": True,
        "font": "Arial",
        "font_size": 12
    },
    "csv": {
        "enabled": True,
        "encoding": "utf-8",
        "separator": ","
    },
    "json": {
        "enabled": True,
        "indent": 2
    }
}

# ============================================================================
# MENSAJES Y TEXTOS
# ============================================================================

MESSAGES = {
    "welcome": "Bienvenido a DemenciaScan - Sistema de Detección de Demencia con IA",
    "model_loaded": "✅ Modelo BERT cargado correctamente",
    "model_not_found": "❌ Modelo no encontrado. Por favor, entrena el modelo primero.",
    "api_configured": "✅ API de Groq configurada",
    "api_not_configured": "⚠️ API de Groq no configurada. Algunas funciones no estarán disponibles.",
    "recording_started": "🎤 Grabación iniciada...",
    "recording_stopped": "⏹️ Grabación detenida",
    "transcription_success": "✅ Transcripción completada",
    "transcription_error": "❌ Error en la transcripción",
    "analysis_started": "🔍 Analizando transcripción...",
    "analysis_complete": "✅ Análisis completado",
    "analysis_error": "❌ Error en el análisis",
    "high_risk": "⚠️ ALTO RIESGO: Se detectaron múltiples indicadores de deterioro cognitivo",
    "moderate_risk": "⚡ RIESGO MODERADO: Algunos indicadores presentes, se recomienda evaluación adicional",
    "low_risk": "✅ BAJO RIESGO: Pocos indicadores de deterioro cognitivo detectados",
    "disclaimer": "⚠️ NOTA: Este es un análisis automatizado preliminar. Para diagnóstico definitivo, consulte con un profesional médico."
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": ROOT_DIR / "logs" / "app.log"
}

# Crear directorio de logs
(ROOT_DIR / "logs").mkdir(exist_ok=True)

# ============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================================

def validate_config():
    """Valida que la configuración esté correcta"""
    issues = []
    
    # Verificar modelo
    if not MODEL_PATH.exists():
        issues.append(f"Modelo no encontrado en: {MODEL_PATH}")
    
    # Verificar API key
    if not GROQ_API_KEY or GROQ_API_KEY == "":
        issues.append("API key de Groq no configurada")
    
    # Verificar directorios
    for dir_name, dir_path in [
        ("DemenciaScan", DEMENCIA_SCAN_DIR),
        ("Models", MODELS_DIR)
    ]:
        if not dir_path.exists():
            issues.append(f"Directorio {dir_name} no encontrado: {dir_path}")
    
    return issues

def print_config_status():
    """Imprime el estado de la configuración"""
    print("=" * 60)
    print("📋 ESTADO DE CONFIGURACIÓN")
    print("=" * 60)
    
    issues = validate_config()
    
    if not issues:
        print("✅ Configuración válida - Todo listo para usar")
    else:
        print("⚠️ Problemas encontrados:")
        for issue in issues:
            print(f"   - {issue}")
    
    print("\n📁 Rutas configuradas:")
    print(f"   Root: {ROOT_DIR}")
    print(f"   Modelo: {MODEL_PATH}")
    print(f"   Datos: {DATA_DIR}")
    print(f"   Exportaciones: {EXPORTS_DIR}")
    
    print("\n🤖 Modelo BERT:")
    print(f"   Tipo: {DEFAULT_BERT_MODEL}")
    print(f"   Nombre: {BERT_MODELS[DEFAULT_BERT_MODEL]}")
    
    print("\n🎤 Audio:")
    print(f"   Sample Rate: {AUDIO_CONFIG['sample_rate']} Hz")
    print(f"   Canales: {AUDIO_CONFIG['channels']}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_config_status()
