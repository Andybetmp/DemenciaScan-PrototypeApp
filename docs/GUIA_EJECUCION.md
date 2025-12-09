# 🚀 GUÍA COMPLETA DE EJECUCIÓN - Proyecto Detección de Demencia

Esta guía te llevará paso a paso para ejecutar el proyecto completo, desde la instalación hasta el uso de la interfaz gráfica.

---

## 📋 TABLA DE CONTENIDOS

1. [Requisitos Previos](#requisitos-previos)
2. [Instalación Paso a Paso](#instalación-paso-a-paso)
3. [Configuración Inicial](#configuración-inicial)
4. [Ejecución del Proyecto](#ejecución-del-proyecto)
5. [Uso de la Interfaz Gráfica](#uso-de-la-interfaz-gráfica)
6. [Solución de Problemas](#solución-de-problemas)

---

## 📌 REQUISITOS PREVIOS

### Hardware Mínimo
- **RAM**: 8GB (16GB recomendado)
- **Almacenamiento**: 5GB libres
- **Procesador**: Intel i5 o equivalente
- **Micrófono**: Para grabación de audio
- **GPU**: Opcional (acelera el entrenamiento BERT)

### Software Requerido
- **Python 3.8 o superior** ([Descargar aquí](https://www.python.org/downloads/))
- **Git** (opcional, para clonar repositorio)
- **Navegador web moderno** (Chrome, Firefox, Edge)

### Verificar Python
```bash
python --version
# Debe mostrar: Python 3.8.x o superior
```

---

## 🔧 INSTALACIÓN PASO A PASO

### Paso 1: Preparar el Entorno

#### Windows:
```bash
# Abrir PowerShell o CMD en la carpeta del proyecto
cd "c:/Users/andyj/Documents/2025 UTP/Prototype app"

# Crear entorno virtual
python -m venv venv_bert

# Activar entorno virtual
venv_bert\Scripts\activate

# Verificar activación (debe aparecer (venv_bert) al inicio de la línea)
```

#### Linux/Mac:
```bash
# Navegar a la carpeta del proyecto
cd "/ruta/al/proyecto"

# Crear entorno virtual
python3 -m venv venv_bert

# Activar entorno virtual
source venv_bert/bin/activate
```

### Paso 2: Instalar Dependencias Base

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias principales
pip install -r requirements.txt

# Esto instalará:
# - PyTorch (modelo BERT)
# - Transformers (Hugging Face)
# - Pandas, NumPy (procesamiento de datos)
# - Scikit-learn (métricas)
# - Matplotlib, Seaborn (visualizaciones)
# - Y más...
```

**⏱️ Tiempo estimado**: 5-10 minutos (dependiendo de la conexión)

### Paso 3: Instalar Dependencias para Interfaz Gráfica

```bash
# Instalar Streamlit y dependencias GUI
pip install streamlit
pip install plotly
pip install streamlit-audio-recorder
pip install fpdf
pip install pillow

# Verificar instalación
streamlit --version
```

### Paso 4: Instalar Dependencias de Audio (para transcripción)

#### Windows:
```bash
pip install pyaudio
pip install keyboard
pip install pyautogui
pip install pyperclip
pip install groq
```

**Nota**: Si PyAudio falla en Windows, descargar el wheel desde:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

```bash
# Instalar wheel descargado
pip install PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl
```

#### Linux:
```bash
# Instalar dependencias del sistema
sudo apt-get install portaudio19-dev python3-pyaudio

# Instalar paquetes Python
pip install pyaudio keyboard pyautogui pyperclip groq
```

#### Mac:
```bash
# Instalar PortAudio
brew install portaudio

# Instalar paquetes Python
pip install pyaudio keyboard pyautogui pyperclip groq
```

### Paso 5: Descargar Modelos de Lenguaje (opcional)

```bash
# Para análisis de texto en español
python -m spacy download es_core_news_sm

# Para análisis multilingüe
python -m spacy download xx_ent_wiki_sm
```

---

## ⚙️ CONFIGURACIÓN INICIAL

### 1. Configurar API de Groq (para transcripción)

1. **Obtener API Key**:
   - Visitar: https://console.groq.com/
   - Crear cuenta gratuita
   - Ir a "API Keys" y crear nueva key
   - Copiar la key (empieza con `gsk_...`)

2. **Configurar en el proyecto**:

Crear archivo `.env` en la raíz del proyecto:
```bash
# Crear archivo .env
echo GROQ_API_KEY=tu_api_key_aqui > .env
```

O editar directamente `dictar.py` (línea 8):
```python
client = Groq(api_key="TU_API_KEY_AQUI")
```

### 2. Verificar Estructura de Archivos

Asegurarse de que existan estos archivos/carpetas:
```
Prototype app/
├── DemenciaScan/
│   ├── models_test/
│   │   └── best_model.pth (modelo entrenado)
│   ├── predict_new_transcription.py
│   ├── train_bert.py
│   └── ...
├── dictar.py
├── requirements.txt
├── GUIA_EJECUCION.md (este archivo)
└── venv_bert/ (entorno virtual)
```

### 3. Entrenar o Descargar Modelo BERT

**Opción A: Usar modelo pre-entrenado** (recomendado)
- Si ya existe `DemenciaScan/models_test/best_model.pth`, ¡listo!

**Opción B: Entrenar desde cero**
```bash
cd DemenciaScan
python main_pipeline.py --num_epochs 20 --batch_size 4
```
⏱️ Tiempo: 30-60 minutos (dependiendo del hardware)

---

## 🎮 EJECUCIÓN DEL PROYECTO

### Método 1: Interfaz Gráfica (RECOMENDADO) 🌟

```bash
# Asegurarse de estar en la raíz del proyecto
cd "c:/Users/andyj/Documents/2025 UTP/Prototype app"

# Activar entorno virtual (si no está activo)
venv_bert\Scripts\activate  # Windows
# source venv_bert/bin/activate  # Linux/Mac

# Ejecutar aplicación web
streamlit run app_streamlit.py
```

**Se abrirá automáticamente en el navegador**: http://localhost:8501

### Método 2: Línea de Comandos (Tradicional)

#### A. Solo Transcripción de Audio
```bash
python dictar.py

# Instrucciones:
# 1. Presionar y mantener tecla INS para grabar
# 2. Soltar INS para detener
# 3. El texto se copia automáticamente al portapapeles
```

#### B. Solo Análisis de Texto
```bash
cd DemenciaScan
python predict_new_transcription.py

# Seguir las instrucciones en pantalla:
# 1. Ingresar ruta del archivo .txt
# 2. Proporcionar datos demográficos
# 3. Ver resultados del análisis
```

#### C. Pipeline Completo (Entrenamiento)
```bash
cd DemenciaScan
python main_pipeline.py

# Ejecuta:
# 1. Preprocesamiento de datos
# 2. Entrenamiento del modelo BERT
# 3. Evaluación y métricas
# 4. Generación de gráficas
```

---

## 🖥️ USO DE LA INTERFAZ GRÁFICA

### Página Principal
- **Información del proyecto**
- **Estado del sistema** (modelo cargado, API configurada)
- **Acceso rápido** a funcionalidades

### 1️⃣ Módulo de Transcripción
1. Hacer clic en "🎤 Transcripción de Audio"
2. Permitir acceso al micrófono (si el navegador lo solicita)
3. Hacer clic en "Iniciar Grabación"
4. Hablar claramente
5. Hacer clic en "Detener Grabación"
6. Ver transcripción generada
7. Opción de guardar o analizar directamente

### 2️⃣ Módulo de Análisis de Texto
1. Hacer clic en "📝 Análisis de Texto"
2. **Opción A**: Cargar archivo .txt
3. **Opción B**: Pegar texto directamente
4. Completar formulario demográfico:
   - Edad del paciente
   - Nivel de educación
   - Calidad de sueño
   - Hábitos diarios
   - Nivel de entorno social
5. Hacer clic en "Analizar"
6. Ver resultados:
   - Predicción (Riesgo/Sin Riesgo)
   - Probabilidades
   - Indicadores clave
   - Visualizaciones interactivas

### 3️⃣ Análisis Completo (Transcripción + Predicción)
1. Hacer clic en "🧠 Análisis Completo"
2. Grabar audio
3. Completar datos demográficos
4. Ver análisis integrado
5. Exportar resultados (PDF/CSV)

### 4️⃣ Historial de Análisis
- Ver análisis previos
- Comparar resultados
- Exportar reportes

### 5️⃣ Configuración
- Cambiar modelo BERT
- Ajustar parámetros
- Configurar API keys
- Preferencias de visualización

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Problema 1: "No module named 'torch'"

**Solución**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problema 2: "CUDA out of memory"

**Solución**:
```bash
# Usar CPU en lugar de GPU
python main_pipeline.py --device cpu --batch_size 2
```

### Problema 3: "API Key de Groq inválida"

**Solución**:
1. Verificar que la API key sea correcta
2. Revisar límites de uso en https://console.groq.com/
3. Generar nueva API key si es necesario

### Problema 4: "No se puede acceder al micrófono"

**Solución Windows**:
1. Configuración → Privacidad → Micrófono
2. Permitir acceso a aplicaciones de escritorio
3. Verificar que el micrófono esté conectado

**Solución Linux**:
```bash
# Verificar dispositivos de audio
arecord -l

# Dar permisos
sudo usermod -a -G audio $USER
```

### Problema 5: "Streamlit no se ejecuta"

**Solución**:
```bash
# Reinstalar Streamlit
pip uninstall streamlit
pip install streamlit

# Verificar instalación
streamlit --version

# Limpiar caché
streamlit cache clear
```

### Problema 6: "Modelo no encontrado"

**Solución**:
```bash
# Verificar que existe el modelo
ls DemenciaScan/models_test/best_model.pth

# Si no existe, entrenar modelo
cd DemenciaScan
python test_training_with_new_data.py
```

### Problema 7: "Error de codificación UTF-8"

**Solución**:
```python
# Al abrir archivos, usar encoding explícito
with open('archivo.txt', 'r', encoding='utf-8') as f:
    contenido = f.read()
```

### Problema 8: "Puerto 8501 ya en uso"

**Solución**:
```bash
# Usar puerto diferente
streamlit run app_streamlit.py --server.port 8502

# O matar proceso existente (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

## 📊 VERIFICACIÓN DE INSTALACIÓN

### Script de Verificación Rápida

Crear archivo `verificar_instalacion.py`:
```python
import sys

def verificar():
    print("🔍 Verificando instalación...\n")
    
    # Python
    print(f"✅ Python: {sys.version}")
    
    # Dependencias principales
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except:
        print("❌ PyTorch no instalado")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except:
        print("❌ Transformers no instalado")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except:
        print("❌ Streamlit no instalado")
    
    try:
        import pyaudio
        print("✅ PyAudio instalado")
    except:
        print("❌ PyAudio no instalado")
    
    try:
        from groq import Groq
        print("✅ Groq instalado")
    except:
        print("❌ Groq no instalado")
    
    print("\n✨ Verificación completa")

if __name__ == "__main__":
    verificar()
```

Ejecutar:
```bash
python verificar_instalacion.py
```

---

## 🎯 FLUJO DE TRABAJO RECOMENDADO

### Para Uso Clínico:

1. **Preparación**:
   - Abrir interfaz gráfica
   - Verificar que el modelo esté cargado
   - Preparar información del paciente

2. **Grabación**:
   - Iniciar sesión de grabación
   - Realizar prueba de descripción de imagen (Cookie Theft)
   - Grabar conversación natural (3-5 minutos)

3. **Análisis**:
   - Revisar transcripción generada
   - Completar datos demográficos
   - Ejecutar análisis BERT

4. **Interpretación**:
   - Revisar probabilidades
   - Analizar indicadores clave
   - Comparar con evaluaciones previas

5. **Documentación**:
   - Exportar resultados
   - Guardar en historial del paciente
   - Compartir con equipo médico

### Para Investigación:

1. **Recolección de Datos**:
   - Grabar múltiples sesiones
   - Etiquetar correctamente
   - Mantener consistencia

2. **Entrenamiento**:
   - Preparar dataset balanceado
   - Ejecutar pipeline completo
   - Validar métricas

3. **Evaluación**:
   - Analizar matriz de confusión
   - Revisar curvas ROC
   - Identificar mejoras

---

## 📚 RECURSOS ADICIONALES

### Documentación
- `README.md`: Visión general del proyecto
- `INICIO_RAPIDO.md`: Guía rápida
- `DemenciaScan/README.md`: Documentación técnica BERT
- `TODO.md`: Estado del proyecto

### Enlaces Útiles
- **Groq API**: https://console.groq.com/docs
- **Streamlit Docs**: https://docs.streamlit.io/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **PyTorch**: https://pytorch.org/docs/

### Soporte
- Revisar issues en el repositorio
- Consultar documentación técnica
- Contactar al equipo de desarrollo

---

## ⚠️ NOTAS IMPORTANTES

### Privacidad y Ética
- ✅ Obtener consentimiento informado antes de grabar
- ✅ Anonimizar datos sensibles
- ✅ Cumplir con regulaciones locales (GDPR, HIPAA, etc.)
- ✅ No usar como único método de diagnóstico

### Limitaciones
- ⚠️ Herramienta de apoyo, no reemplazo de evaluación clínica
- ⚠️ Requiere validación con profesionales médicos
- ⚠️ Resultados pueden variar según calidad de audio
- ⚠️ Modelo entrenado con datos específicos

### Mejores Prácticas
- 🎯 Usar en ambiente silencioso
- 🎯 Micrófono de buena calidad
- 🎯 Hablar claramente
- 🎯 Sesiones de 3-5 minutos
- 🎯 Revisar transcripción antes de analizar

---

## 🚀 PRÓXIMOS PASOS

Una vez que el sistema esté funcionando:

1. **Familiarízate con la interfaz**
2. **Prueba con ejemplos de transcripciones**
3. **Ajusta parámetros según necesidades**
4. **Documenta casos de uso**
5. **Proporciona feedback para mejoras**

---

**¡Listo para comenzar! 🧠🤖**

Si tienes problemas, revisa la sección de [Solución de Problemas](#solución-de-problemas) o consulta la documentación adicional.
