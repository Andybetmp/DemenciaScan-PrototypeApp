# 🚀 INICIO RÁPIDO - BERT Multilingüe para Detección de Demencia

## ⚡ Instalación en 3 Pasos

### 1. Ejecutar Setup Automático
```bash
python setup_bert.py
```

### 2. Activar Entorno Virtual
```bash
# Windows
venv_bert\Scripts\activate

# Linux/Mac
source venv_bert/bin/activate
```

### 3. Ejecutar Pipeline
```bash
cd DemenciaScan
python main_pipeline.py
```

## 🎯 ¿Qué hace cada paso?

### Setup Automático (`setup_bert.py`)
- ✅ Verifica Python 3.8+
- ✅ Crea entorno virtual `venv_bert`
- ✅ Instala todas las dependencias
- ✅ Verifica que todo funcione correctamente

### Pipeline Principal (`main_pipeline.py`)
- 📊 **Preprocesamiento**: Tokeniza texto, detecta idiomas, extrae características
- 🤖 **Entrenamiento**: Fine-tuning de BERT multilingüe (20 épocas)
- 📈 **Evaluación**: Métricas, gráficas, análisis de resultados

## 📊 Resultados Esperados

Después de ejecutar el pipeline encontrarás:

```
DemenciaScan/
├── models/
│   ├── best_model.pth              # Modelo entrenado
│   ├── training_history.png        # Gráficas de entrenamiento
│   └── evaluation_results/         # Métricas y visualizaciones
│       ├── confusion_matrix_test.png
│       ├── roc_curve_test.png
│       └── metrics_test.json
```

## ⚙️ Configuración Personalizada

### Cambiar Modelo BERT
```bash
# Español
python main_pipeline.py --model_type spanish

# Inglés  
python main_pipeline.py --model_type english

# Multilingüe (default)
python main_pipeline.py --model_type multilingual
```

### Ajustar Entrenamiento
```bash
# Más épocas para mejor rendimiento
python main_pipeline.py --num_epochs 50

# Batch size más grande (si tienes GPU potente)
python main_pipeline.py --batch_size 8

# Learning rate personalizado
python main_pipeline.py --learning_rate 1e-5
```

## 🔧 Solución de Problemas Rápida

### Error: "No module named 'torch'"
```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: "CUDA out of memory"
```bash
# Reducir batch size
python main_pipeline.py --batch_size 2
```

### Error: "No se encontró transcripciones_procesadas.csv"
```bash
# Verificar que el archivo esté en el directorio correcto
ls transcripciones_procesadas.csv
```

## 📈 Interpretación de Resultados

### Métricas Clave
- **Accuracy > 0.8**: Buen rendimiento general
- **F1-Score > 0.7**: Balance adecuado precisión/recall
- **AUC-ROC > 0.8**: Buena capacidad de discriminación

### Qué Buscar
- 📊 **Curva de entrenamiento**: Debe converger sin overfitting
- 🎯 **Matriz de confusión**: Pocos falsos positivos/negativos
- 🔍 **Importancia de características**: Qué variables son más predictivas

## 🚀 Próximos Pasos

### 1. Expandir Dataset
- Agregar más transcripciones
- Balancear clases (Alzheimer vs No Alzheimer)
- Incluir más variedad demográfica

### 2. Optimizar Modelo
- Probar diferentes hiperparámetros
- Implementar ensemble de modelos
- Usar técnicas de data augmentation

### 3. Análisis Avanzado
- Estudiar casos mal clasificados
- Analizar patrones lingüísticos específicos
- Validar con datos clínicos reales

## 📚 Documentación Completa

- 📖 **README.md**: Documentación técnica completa
- 📋 **TODO.md**: Estado del proyecto y próximos pasos
- 🔧 **Código fuente**: Comentarios detallados en cada archivo

## 🆘 Soporte

Si encuentras problemas:
1. Revisa los mensajes de error en la consola
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de tener suficiente memoria RAM (mínimo 8GB recomendado)
4. Consulta la documentación técnica en README.md

---

**¡En menos de 5 minutos tendrás tu modelo BERT funcionando! 🧠🤖**
