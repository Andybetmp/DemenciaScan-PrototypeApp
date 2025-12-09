# 🧠 Prototype App - Detección de Demencia con IA

Esta aplicación prototipo combina transcripción de audio en tiempo real y análisis de texto con modelos BERT multilingües para detectar posibles signos de demencia en conversaciones.

## 📋 Componentes de la Aplicación

### 1. **DemenciaScan** - Análisis BERT
- Ubicación: `src/DemenciaScan`
- Modelo BERT multilingüe para detección de demencia
- Análisis de transcripciones y características demográficas
- Pipeline completo: preprocesamiento, entrenamiento y evaluación

### 2. **dictar.py** - Transcripción de Audio
- Ubicación: `src/dictar.py`
- Grabación de audio en tiempo real
- Transcripción automática con Groq Whisper
- Copia automática al portapapeles

### 3. **INT/app** - Interfaz de Transcripción
- Ubicación: `src/INT/app`
- Versión alternativa del sistema de transcripción
- Grabación y transcripción integrada

## 🔧 Requisitos del Sistema

### Hardware Mínimo
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: 5GB libres
- **Procesador**: CPU moderna (Intel i5 o superior)
- **GPU**: Opcional pero recomendado para entrenamiento BERT

### Software Requerido
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **Micrófono**: Para funciones de grabación de audio

### Dependencias Python
- PyTorch y Transformers (para BERT)
- PyAudio (para grabación de audio)
- Groq API (para transcripción)
- Bibliotecas de procesamiento de datos

## 🚀 Instalación Completa

### Paso 1: Clonar o Descargar el Proyecto

```bash
# Si usas Git
git clone <url-del-repositorio>
cd "2025 UTP/Prototype app"
```

### Paso 2: Instalar Python y Crear Entorno Virtual

```bash
# Verificar versión de Python
python --version
# Debe ser 3.8 o superior

# Crear entorno virtual
python -m venv venv_bert

# Activar entorno virtual
# Windows:
venv_bert\Scripts\activate
# Linux/Mac:
source venv_bert/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

### Paso 4: Configurar API de Groq (para transcripción)

1. Obtener API key de [Groq Console](https://console.groq.com/)
2. Reemplazar la API key en los archivos:
   - `src/dictar.py` (línea con `api_key=`)
   - `src/INT/app/app.py` (línea con `api_key=`)
   - O mejor aún, crear un archivo `.env` en la raíz con `GROQ_API_KEY=tu_api_key`

### Paso 5: Verificar Instalación

```bash
# Ejecutar script de verificación
python scripts/setup_bert.py
```

Este script:
- ✅ Verifica Python 3.8+
- ✅ Crea entorno virtual si no existe
- ✅ Instala dependencias faltantes
- ✅ Verifica funcionamiento básico

## 📊 Uso de la Aplicación

### Flujo de Trabajo General

1. **Grabación**: Usar `src/dictar.py` para grabar conversaciones
2. **Transcripción**: Audio se convierte automáticamente a texto
3. **Análisis**: `DemenciaScan` procesa el texto con BERT
4. **Resultado**: Predicción de posibles signos de demencia

### Interfaz Gráfica (Recomendado)

```bash
streamlit run src/app_streamlit.py
```

### Componente 1: DemenciaScan (Análisis BERT)

#### Instalación Específica
```bash
# Navegar al directorio
cd src/DemenciaScan

# Ejecutar pipeline completo
python main_pipeline.py
```

#### Archivos de Datos Requeridos
- `data/transcripciones_procesadas.csv`: Dataset con transcripciones y metadatos
- Columnas requeridas: `texto`, `edad`, `nivel_educacion`, `diagnostico_alzheimer`, etc.

#### Comandos Principales
```bash
# Pipeline completo
python src/DemenciaScan/main_pipeline.py --num_epochs 20 --batch_size 4

# Solo preprocesamiento
python src/DemenciaScan/data_preprocessing.py

# Solo entrenamiento
python src/DemenciaScan/train_bert.py

# Solo evaluación
python src/DemenciaScan/evaluate_model.py
```

### Componente 2: dictar.py (Transcripción)

#### Uso Básico
```bash
# Ejecutar transcriptor
python src/dictar.py
```

#### Instrucciones de Uso
1. Presiona la tecla **INS** para comenzar a grabar
2. Mantén presionada **INS** mientras hablas
3. Suelta **INS** para detener la grabación
4. El texto se transcribe automáticamente y se copia al portapapeles

#### Características
- Transcripción en español con atención a anomalías del habla
- Detección de pausas, repeticiones y autocorrecciones
- Optimizado para análisis clínico de demencia

### Componente 3: INT/app (Interfaz Alternativa)

#### Uso
```bash
# Navegar al directorio
cd src/INT/app

# Ejecutar aplicación
python app.py
```

*Nota: Esta versión está en desarrollo y puede tener funcionalidades limitadas.*

## ⚙️ Configuración Avanzada

### Parámetros de BERT
```bash
# Modelo multilingüe (recomendado)
python src/DemenciaScan/main_pipeline.py --model_type multilingual

# Más épocas para mejor entrenamiento
python src/DemenciaScan/main_pipeline.py --num_epochs 50 --batch_size 8

# Learning rate personalizado
python src/DemenciaScan/main_pipeline.py --learning_rate 1e-5
```

### Configuración de Audio
- **Frecuencia de muestreo**: 16000 Hz (configurable en código)
- **Canales**: Mono (1 canal)
- **Formato**: 16-bit PCM

## 🔍 Cómo Funciona la Aplicación

### Arquitectura General

```
Audio Input → Transcripción (Whisper) → Análisis BERT → Predicción
     ↓              ↓                        ↓           ↓
  Grabación    Texto + Metadatos        Características  Resultado
  en tiempo                         Lingüísticas +     de Demencia
  real                              Demográficas
```

### Proceso Detallado

1. **Captura de Audio**: `dictar.py` graba audio usando PyAudio
2. **Transcripción**: Groq Whisper convierte audio a texto con prompts clínicos
3. **Preprocesamiento**: `DemenciaScan` tokeniza texto y extrae características
4. **Modelo BERT**: Analiza patrones lingüísticos y demográficos
5. **Predicción**: Clasifica como posible demencia o no

### Características Analizadas
- **Lingüísticas**: Riqueza léxica, repeticiones, pausas, errores
- **Demográficas**: Edad, educación, calidad de sueño, hábitos sociales
- **Patrones**: Estructura de la conversación, coherencia

## 📈 Resultados Esperados

### Métricas de Rendimiento
- **Accuracy**: >80% en datasets balanceados
- **F1-Score**: >70% para balance precisión/recall
- **AUC-ROC**: >80% para discriminación

### Salidas del Sistema
- Modelo entrenado (`data/models/best_model.pth`)
- Gráficas de entrenamiento
- Matriz de confusión y curvas ROC
- Análisis de importancia de características

## 🔧 Solución de Problemas

### Problemas Comunes

#### "No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### "Error de CUDA"
```bash
# Usar CPU en lugar de GPU
python src/DemenciaScan/main_pipeline.py --device cpu
```

#### "API Key de Groq inválida"
- Verificar que la API key esté correcta
- Revisar límites de uso en Groq Console

#### "No se puede acceder al micrófono"
- Verificar permisos de micrófono en el sistema
- Probar con otro dispositivo de audio

#### "Archivo CSV no encontrado"
```bash
# Verificar ubicación del archivo
ls data/transcripciones_procesadas.csv
```

### Verificación de Instalación
```bash
# Probar importaciones básicas
python -c "import torch, transformers, pyaudio; print('✅ Dependencias OK')"
```

## 📚 Documentación Adicional

- `docs/INICIO_RAPIDO.md`: Guía rápida para principiantes
- `src/DemenciaScan/README.md`: Documentación técnica detallada
- `docs/TODO.md`: Estado del proyecto y tareas pendientes
- `venv_activation_instructions.txt`: Instrucciones de entorno virtual

## 🚀 Próximos Pasos

### Mejoras Planeadas
1. **Interfaz Gráfica**: Aplicación web con Streamlit
2. **Base de Datos**: Almacenamiento persistente de transcripciones
3. **API REST**: Servicio web para integración
4. **Modelos Avanzados**: Ensemble de múltiples BERT
5. **Análisis en Tiempo Real**: Procesamiento continuo

### Expansión del Dataset
- Más transcripciones clínicas
- Variedad demográfica
- Diferentes tipos de demencia

## 🤝 Contribuciones

Para contribuir:
1. Reportar bugs o sugerir mejoras
2. Proponer nuevas características
3. Ayudar con documentación

## 📄 Licencia

Este proyecto es para fines educativos e investigativos. Consultar con instituciones médicas antes de uso clínico.

## ⚠️ Advertencias Importantes

- **No es un diagnóstico médico**: Esta herramienta es experimental
- **Privacidad**: Manejar datos sensibles con cuidado
- **Ética**: Usar solo con consentimiento informado
- **Validación**: Resultados deben ser validados por profesionales

---

**¡Listo para explorar la detección de demencia con IA! 🧠🤖**
