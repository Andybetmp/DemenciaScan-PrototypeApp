# 🧠 BERT Multilingüe para Detección de Demencia

Este proyecto implementa un modelo BERT multilingüe (español/inglés) para la detección de demencia basada en transcripciones de audio y características demográficas.

## 📋 Características Principales

- **Modelo BERT Multilingüe**: Soporte para español e inglés con detección automática de idioma
- **Fusión Multimodal**: Combina características textuales y demográficas
- **Arquitectura Avanzada**: Attention mechanisms, Focal Loss, y modelos Ensemble
- **Pipeline Completo**: Preprocesamiento, entrenamiento, evaluación y visualización
- **Métricas Detalladas**: ROC, Precision-Recall, matriz de confusión, importancia de características

## 🚀 Instalación y Configuración

### Paso 1: Preparar el Entorno

```bash
# Crear entorno virtual (recomendado)
python -m venv venv_bert

# Activar entorno virtual
# En Windows:
venv_bert\Scripts\activate
# En Linux/Mac:
source venv_bert/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Verificar Datos

Asegúrate de tener el archivo `transcripciones_procesadas.csv` en el directorio padre con las siguientes columnas:
- `texto`: Transcripción del audio
- `edad`: Edad del paciente
- `nivel_educacion`: Nivel educativo
- `calidad_sueno`: Calidad del sueño
- `habitos_diarios`: Hábitos diarios
- `nivel_entorno_social`: Nivel de entorno social
- `diagnostico_alzheimer`: Diagnóstico (si/no)

## 🔧 Uso del Sistema

### Opción 1: Pipeline Completo (Recomendado)

```bash
# Ejecutar pipeline completo
python main_pipeline.py

# Con parámetros personalizados
python main_pipeline.py --num_epochs 30 --batch_size 8 --learning_rate 1e-5
```

### Opción 2: Ejecución Paso a Paso

#### 1. Preprocesamiento
```bash
python data_preprocessing.py
```

#### 2. Entrenamiento
```bash
python train_bert.py
```

#### 3. Evaluación
```bash
python evaluate_model.py
```

## ⚙️ Configuración Avanzada

### Parámetros del Pipeline

```bash
python main_pipeline.py --help
```

Parámetros principales:
- `--model_type`: Tipo de modelo (`multilingual`, `spanish`, `english`)
- `--num_epochs`: Número de épocas (default: 20)
- `--batch_size`: Tamaño del batch (default: 4)
- `--learning_rate`: Tasa de aprendizaje (default: 2e-5)
- `--max_length`: Longitud máxima de secuencia (default: 512)

### Modelos Disponibles

1. **Multilingüe** (Recomendado):
   ```bash
   python main_pipeline.py --model_type multilingual
   ```

2. **Español**:
   ```bash
   python main_pipeline.py --model_type spanish
   ```

3. **Inglés**:
   ```bash
   python main_pipeline.py --model_type english
   ```

## 📊 Resultados y Visualizaciones

Después de ejecutar el pipeline, encontrarás:

### Directorio `models/`
- `best_model.pth`: Mejor modelo entrenado
- `training_history.png`: Gráficas de entrenamiento
- `pipeline_config.json`: Configuración utilizada

### Directorio `models/evaluation_results/`
- `metrics_test.json`: Métricas del conjunto de prueba
- `confusion_matrix_test.png`: Matriz de confusión
- `roc_curve_test.png`: Curva ROC
- `pr_curve_test.png`: Curva Precision-Recall

## 🧪 Ejemplo de Uso Programático

```python
from data_preprocessing import DemenciaPreprocessor
from bert_model import MultimodalBERTClassifier, create_model_config
from train_bert import DemenciaTrainer

# 1. Preprocesar datos
preprocessor = DemenciaPreprocessor()
data = preprocessor.prepare_data('transcripciones_procesadas.csv')
dataloaders = preprocessor.create_dataloaders(data)

# 2. Crear y entrenar modelo
config = create_model_config('multilingual')
trainer = DemenciaTrainer(config)
history = trainer.train(dataloaders['train'], dataloaders['val'])

# 3. Evaluar modelo
from evaluate_model import DemenciaEvaluator
evaluator = DemenciaEvaluator('models/best_model.pth')
results = evaluator.evaluate_dataset(dataloaders['test'])
```

## 📈 Interpretación de Resultados

### Métricas Principales
- **Accuracy**: Precisión general del modelo
- **F1-Score**: Balance entre precisión y recall
- **AUC-ROC**: Capacidad de discriminación del modelo
- **Precision/Recall**: Por clase (Alzheimer vs No Alzheimer)

### Características Importantes
El modelo analiza automáticamente qué características son más importantes:
- Métricas lingüísticas (riqueza léxica, repeticiones)
- Características demográficas (edad, educación)
- Patrones de habla (pausas, preguntas, exclamaciones)

## 🔍 Solución de Problemas

### Error: "No module named 'transformers'"
```bash
pip install transformers torch
```

### Error: "CUDA out of memory"
```bash
# Reducir batch size
python main_pipeline.py --batch_size 2
```

### Error: "No se encontró el archivo CSV"
```bash
# Verificar ruta del archivo
python main_pipeline.py --csv_path ruta/correcta/transcripciones_procesadas.csv
```

### Advertencia: "Dataset muy pequeño"
- El modelo funciona mejor con más datos
- Considera expandir el dataset con más transcripciones
- Usa técnicas de data augmentation

## 🚀 Mejoras Futuras

### Expansión del Dataset
1. Agregar más transcripciones
2. Incluir diferentes tipos de demencia
3. Balancear clases (Alzheimer vs No Alzheimer)

### Optimizaciones del Modelo
1. Probar diferentes arquitecturas BERT
2. Implementar data augmentation
3. Usar técnicas de regularización avanzadas

### Características Adicionales
1. Análisis de prosodia (tono, ritmo)
2. Características temporales del habla
3. Integración con datos de neuroimagen

## 📚 Referencias Técnicas

### Modelos BERT Utilizados
- **mBERT**: `bert-base-multilingual-cased`
- **BETO**: `dccuchile/bert-base-spanish-wwm-cased`
- **DistilBERT**: `distilbert-base-uncased`

### Arquitectura del Modelo
```
Input Text → BERT Encoder → Attention Layer
                                ↓
Additional Features → Dense Layers → Fusion Layer → Classifier
```

### Función de Pérdida
- **Focal Loss**: Para manejar desbalance de clases
- **CrossEntropy**: Alternativa estándar

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Implementa mejoras
4. Envía un pull request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para detalles.

## 📞 Soporte

Para preguntas o problemas:
1. Revisa la sección de solución de problemas
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de tener suficiente memoria RAM/GPU

---

**¡Listo para detectar demencia con IA! 🧠🤖**
