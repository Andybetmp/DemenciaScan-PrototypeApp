# 🚀 INICIO RÁPIDO - Interfaz Gráfica DemenciaScan

## ⚡ Ejecutar en 3 Pasos

### 1️⃣ Instalar Dependencias GUI
```bash
# Activar entorno virtual (si no está activo)
venv_bert\Scripts\activate  # Windows
# source venv_bert/bin/activate  # Linux/Mac

# Instalar dependencias de la interfaz gráfica
pip install -r requirements_gui.txt
```

### 2️⃣ Verificar Modelo BERT
```bash
# Verificar que existe el modelo entrenado
# Debe estar en: DemenciaScan/models_test/best_model.pth

# Si no existe, entrenar el modelo:
cd DemenciaScan
python test_training_with_new_data.py
cd ..
```

### 3️⃣ Ejecutar Aplicación Web
```bash
# Ejecutar Streamlit
streamlit run app_streamlit.py

# La aplicación se abrirá automáticamente en:
# http://localhost:8501
```

---

## 🎯 Funcionalidades Disponibles

### 📝 Análisis de Texto
1. Ir a "📝 Análisis de Texto" en el menú lateral
2. Cargar archivo .txt, pegar texto o usar ejemplo
3. Completar información demográfica
4. Hacer clic en "🔍 Analizar Texto"
5. Ver resultados con visualizaciones interactivas
6. Exportar resultados (JSON, CSV, PDF)

### 🎤 Transcripción de Audio
1. Ejecutar `dictar.py` en una terminal separada:
   ```bash
   python dictar.py
   ```
2. Presionar y mantener tecla INS para grabar
3. Soltar INS para detener
4. Copiar transcripción generada
5. Pegar en la interfaz web para análisis

### 📊 Historial
- Ver todos los análisis realizados
- Gráfico de evolución temporal
- Comparar resultados
- Limpiar historial

### ⚙️ Configuración
- Ajustar umbrales de clasificación
- Configurar API de Groq
- Ver información del sistema

---

## 🖥️ Interfaz de Usuario

### Navegación Principal
```
🏠 Inicio              → Información general y estadísticas
📝 Análisis de Texto   → Analizar transcripciones
🎤 Transcripción       → Instrucciones para grabar audio
🧠 Análisis Completo   → Flujo integrado (en desarrollo)
📊 Historial           → Ver análisis previos
⚙️ Configuración       → Ajustes del sistema
ℹ️ Ayuda               → Documentación y FAQ
```

### Visualizaciones Incluidas
- 📊 **Gauge de Probabilidad**: Medidor visual del riesgo
- 📈 **Gráfico de Barras**: Comparación de probabilidades
- 🕸️ **Gráfico de Radar**: Indicadores de demencia
- 📉 **Línea Temporal**: Evolución de análisis

---

## 📋 Ejemplo de Uso Completo

### Caso 1: Análisis con Transcripción Existente

```bash
# 1. Iniciar aplicación
streamlit run app_streamlit.py

# 2. En el navegador:
#    - Ir a "📝 Análisis de Texto"
#    - Seleccionar "📋 Usar ejemplo"
#    - Completar datos demográficos:
#      * Edad: 75
#      * Educación: Secundaria
#      * Sueño: Regular
#      * Hábitos: Sedentario
#      * Entorno social: Medio
#    - Clic en "🔍 Analizar Texto"
#    - Ver resultados y visualizaciones
#    - Exportar si es necesario
```

### Caso 2: Análisis con Nueva Grabación

```bash
# Terminal 1: Ejecutar dictar.py
python dictar.py
# Presionar INS, hablar, soltar INS
# Transcripción se copia al portapapeles

# Terminal 2: Aplicación web ya corriendo
streamlit run app_streamlit.py

# En el navegador:
#    - Ir a "📝 Análisis de Texto"
#    - Seleccionar "✍️ Pegar texto directamente"
#    - Pegar transcripción (Ctrl+V)
#    - Completar datos demográficos
#    - Analizar y ver resultados
```

---

## 🔧 Solución de Problemas Rápida

### Error: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Error: "Modelo no encontrado"
```bash
# Entrenar modelo
cd DemenciaScan
python test_training_with_new_data.py
```

### Error: "Puerto 8501 ya en uso"
```bash
# Usar puerto diferente
streamlit run app_streamlit.py --server.port 8502
```

### Error: "Cannot import DemenciaPredictor"
```bash
# Verificar que estás en el directorio correcto
cd "c:/Users/andyj/Documents/2025 UTP/Prototype app"
streamlit run app_streamlit.py
```

### La aplicación no se abre automáticamente
```
Abrir manualmente en el navegador:
http://localhost:8501
```

---

## 📊 Interpretación de Resultados

### Niveles de Riesgo

| Probabilidad | Nivel | Color | Interpretación |
|--------------|-------|-------|----------------|
| < 50% | 🟢 Bajo Riesgo | Verde | Pocos indicadores detectados |
| 50-70% | 🟡 Riesgo Moderado | Amarillo | Algunos indicadores presentes |
| > 70% | 🔴 Alto Riesgo | Rojo | Múltiples indicadores detectados |

### Indicadores Clave

- **Problemas de Comprensión**: Dificultad para entender instrucciones
- **Patrones Repetitivos**: Repetición de palabras o frases
- **Muletillas**: Uso excesivo de pausas y rellenos
- **Errores Semánticos**: Confusión de significados
- **Errores Fonémicos**: Problemas de pronunciación

---

## 💾 Exportación de Resultados

### Formatos Disponibles

1. **JSON** (`.json`)
   - Datos estructurados completos
   - Ideal para procesamiento posterior
   - Incluye todos los metadatos

2. **CSV** (`.csv`)
   - Formato tabular
   - Fácil de importar en Excel
   - Bueno para análisis estadístico

3. **PDF** (`.pdf`)
   - Reporte visual completo
   - Listo para imprimir
   - Incluye interpretación

### Ubicación de Archivos
```
Prototype app/
└── exports/
    ├── analisis_20250115_143022.json
    ├── analisis_20250115_143022.csv
    └── reporte_20250115_143022.pdf
```

---

## 🎨 Personalización

### Cambiar Umbrales de Riesgo
1. Ir a "⚙️ Configuración"
2. Ajustar sliders:
   - Umbral de Alto Riesgo (default: 0.70)
   - Umbral de Riesgo Moderado (default: 0.50)
3. Los cambios se aplican inmediatamente

### Configurar API de Groq
1. Obtener API key en https://console.groq.com/
2. Ir a "⚙️ Configuración"
3. Ingresar API key
4. Guardar (o editar archivo `.env`)

---

## 📚 Recursos Adicionales

### Documentación
- 📖 **GUIA_EJECUCION.md**: Guía completa y detallada
- 📋 **README.md**: Visión general del proyecto
- 🔧 **TODO_GUI.md**: Estado de desarrollo

### Soporte
- Revisar sección "ℹ️ Ayuda" en la aplicación
- Consultar FAQ integrada
- Verificar logs en `logs/app.log`

---

## ⚠️ Notas Importantes

### Privacidad y Seguridad
- ✅ Los datos se procesan localmente
- ✅ No se envían datos a servidores externos (excepto API de Groq para transcripción)
- ✅ Historial se guarda localmente en `exports/history.json`
- ⚠️ Manejar datos sensibles con cuidado

### Limitaciones
- 🔴 **NO es un diagnóstico médico**
- 🔴 Requiere interpretación profesional
- 🔴 Resultados pueden variar según calidad de audio
- 🔴 Modelo entrenado con datos específicos

### Mejores Prácticas
- 🎯 Usar en ambiente silencioso
- 🎯 Micrófono de buena calidad
- 🎯 Sesiones de 3-5 minutos
- 🎯 Revisar transcripción antes de analizar
- 🎯 Guardar resultados importantes

---

## 🚀 ¡Listo para Usar!

La interfaz gráfica está completamente funcional. Sigue los pasos anteriores y comienza a analizar transcripciones.

**Comando rápido para iniciar:**
```bash
streamlit run app_streamlit.py
```

**¡Disfruta de DemenciaScan! 🧠🤖**

---

## 📞 Contacto y Soporte

Si encuentras problemas:
1. Revisa la sección de solución de problemas
2. Consulta `GUIA_EJECUCION.md` para detalles
3. Verifica logs en `logs/app.log`
4. Contacta al equipo de desarrollo

**Versión**: 1.0.0  
**Última actualización**: Enero 2025
