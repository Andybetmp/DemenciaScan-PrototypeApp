from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.aws.ml import SagemakerModel  # Cambiado para usar AWS ML
from diagrams.custom import Custom
from diagrams.generic.device import Mobile
from diagrams.generic.storage import Storage
from diagrams.onprem.database import PostgreSQL

# ============================================================================
# DIAGRAMA PRINCIPAL: ARQUITECTURA LOCAL DE DEMENCIASCANAPP
# ============================================================================

with Diagram("Arquitectura Técnica de DemenciaScanApp - Versión Local", show=False, direction="LR") as diag_local:

    # Usuario final
    usuario = User("Usuario Final")
    # # Representa al paciente o clínico que interactúa con la aplicación para análisis de voz/texto

    with Cluster("Capa de Interfaz de Usuario (UI)"):
        # # Esta capa maneja la interacción del usuario, proporcionando una interfaz intuitiva para entrada de datos y visualización de resultados
        streamlit_app = Python("Streamlit App")
        # # Aplicación web construida con Streamlit que permite grabar voz, cargar archivos de audio, ingresar texto y mostrar resultados con gráficos interactivos

        with Cluster("Paginas de la UI"):
            pagina_inicio = Custom("Pagina Inicio", "./icons/home.png")  # # Página principal con descripción del sistema y navegación
            pagina_analisis = Custom("Analisis de Texto", "./icons/text_analysis.png")  # # Interfaz para análisis de transcripciones existentes con formularios demográficos
            pagina_transcripcion = Custom("Transcripcion", "./icons/microphone.png")  # # Módulo para grabación de audio y conversión a texto
            pagina_completo = Custom("Analisis Completo", "./icons/integration.png")  # # Flujo integrado de transcripción + análisis
            pagina_historial = Custom("Historial", "./icons/history.png")  # # Visualización de análisis previos y comparación de resultados
            pagina_config = Custom("Configuracion", "./icons/settings.png")  # # Ajustes del sistema, umbrales y configuración de API

    with Cluster("Capa de Procesamiento y Modelo IA"):
        # # Esta capa procesa los datos de entrada, extrae características lingüísticas y aplica modelos de IA para clasificación de riesgo de demencia
        preprocesamiento = Python("Modulos de Preprocesamiento")
        # # Incluye data_preprocessing.py: Limpieza de texto, normalización, tokenización y preparación de datos para el modelo

        extraccion_features = Python("Extraccion de Caracteristicas")
        # # Análisis lingüístico: riqueza léxica, repeticiones, pausas, errores semánticos y otros indicadores de deterioro cognitivo

        modelo_bert = Custom("Modelo BERT Fine-tuned", "./icons/bert.png")
        # # Modelo de lenguaje BERT multilingüe entrenado para clasificación lingüística, que predice probabilidad de riesgo de demencia basado en patrones de habla

        predictor = Python("DemenciaPredictor")
        # # Clase principal (predict_new_transcription.py) que integra el modelo BERT con datos demográficos para generar predicciones

    with Cluster("Capa de Servicios Externos"):
        # # Servicios externos utilizados para conversión de voz a texto, accesibles vía API REST
        groq_whisper = Custom("Groq Whisper API", "./icons/groq.png")
        # # API de Groq para transcripción automática de audio a texto, optimizada para capturar pausas, repeticiones y anomalías en el habla

        google_speech = Custom("Google SpeechRecognition", "./icons/google_cloud.png")  # # Alternativa opcional: API de Google para reconocimiento de voz

    with Cluster("Capa de Almacenamiento y Resultados"):
        # # Almacenamiento local de datos, resultados y logs para persistencia y análisis posterior
        archivos_locales = Storage("Archivos Locales")
        # # Almacenamiento de transcripciones, resultados de análisis en formatos CSV, JSON y PDF

        historial = Storage("Historial de Analisis")
        # # Archivo local (JSON) para guardar historial de predicciones, permitiendo comparación temporal y seguimiento

        logs = Storage("Logs del Sistema")
        # # Registros locales de operaciones, errores y métricas de rendimiento para debugging y auditoría

    with Cluster("Capa de Visualizacion"):
        # # Generación de gráficos y reportes visuales para interpretación de resultados
        plotly_charts = Custom("Plotly Charts", "./icons/plotly.png")  # # Gráficos interactivos: gauge de probabilidad, barras de comparación, radar de indicadores
        matplotlib_plots = Custom("Matplotlib Plots", "./icons/matplotlib.png")  # # Gráficos estáticos adicionales si es necesario
        export_reports = Python("Exportacion de Reportes")
        # # Generación de PDFs con resultados detallados, incluyendo interpretaciones y disclaimers médicos

    # Flujo de datos: Entrada del usuario
    usuario >> Edge(label="Graba voz / Carga audio / Ingresa texto") >> streamlit_app

    # Dentro de UI
    streamlit_app >> Edge(label="Navegacion") >> [pagina_inicio, pagina_analisis, pagina_transcripcion, pagina_completo, pagina_historial, pagina_config]

    # De UI a Servicios Externos (para transcripción)
    pagina_transcripcion >> Edge(label="Audio grabado") >> groq_whisper
    groq_whisper >> Edge(label="Transcripcion generada") >> pagina_transcripcion

    # De UI a Procesamiento (para análisis directo de texto)
    pagina_analisis >> Edge(label="Texto + Datos demograficos") >> preprocesamiento

    # Procesamiento interno
    preprocesamiento >> Edge(label="Texto limpio") >> extraccion_features
    extraccion_features >> Edge(label="Caracteristicas language") >> modelo_bert
    modelo_bert >> Edge(label="Predicciones de riesgo") >> predictor

    # De Servicios Externos a Procesamiento (flujo completo)
    groq_whisper >> Edge(label="Transcripcion") >> preprocesamiento

    # De Procesamiento a Visualización
    predictor >> Edge(label="Resultados de analisis") >> plotly_charts
    plotly_charts >> Edge(label="Graficos generados") >> streamlit_app

    # De Procesamiento a Almacenamiento
    predictor >> Edge(label="Resultados") >> archivos_locales
    predictor >> Edge(label="Logs") >> logs
    streamlit_app >> Edge(label="Historial") >> historial

    # De Visualización a Exportación
    plotly_charts >> Edge(label="Datos para reporte") >> export_reports
    export_reports >> Edge(label="PDF/CSV/JSON") >> archivos_locales

    # Salida al usuario
    streamlit_app >> Edge(label="Resultados visualizados") >> usuario

# ============================================================================
# DIAGRAMA OPCIONAL: ARQUITECTURA EN NUBE (AWS/GCP)
# ============================================================================

with Diagram("Arquitectura Técnica de DemenciaScanApp - Versión Cloud (AWS)", show=False, direction="LR") as diag_cloud:

    # Usuario final
    usuario_cloud = User("Usuario Final")

    with Cluster("Capa de Interfaz de Usuario (UI) - Cloud"):
        # # Interfaz desplegada en servicios cloud para mayor escalabilidad y acceso remoto
        streamlit_cloud = Python("Streamlit App en EC2")
        # # Instancia EC2 ejecutando la aplicación Streamlit, accesible vía HTTPS con balanceo de carga

    with Cluster("Capa de Procesamiento y Modelo IA - Cloud"):
        # # Procesamiento distribuido en la nube con servicios de ML gestionados
        lambda_preprocessing = Python("AWS Lambda - Preprocesamiento")
        # # Función serverless para limpieza y extracción de características, escalable bajo demanda

        sagemaker_model = SagemakerModel("SageMaker Endpoint - BERT")
        # # Endpoint de SageMaker alojando el modelo BERT fine-tuned, con auto-scaling para predicciones

    with Cluster("Capa de Servicios Externos - Cloud"):
        # # APIs externas accesibles vía endpoints HTTPS en la nube
        groq_api_cloud = Custom("Groq Whisper API", "./icons/groq.png")
        # # API de Groq integrada mediante llamadas HTTPS desde la nube

    with Cluster("Capa de Almacenamiento y Resultados - Cloud"):
        # # Almacenamiento cloud para datos persistentes y backups
        s3_storage = Storage("local data base")
        # # Bucket S3 para almacenar transcripciones, resultados en CSV/JSON/PDF y backups

        cloudwatch_logs = Custom("CloudWatch Logs", "./icons/cloudwatch.png")
        # # Servicio de logging y monitoreo para rastrear operaciones y errores en tiempo real

    with Cluster("Capa de Visualizacion - Cloud"):
        # # Generación de visualizaciones en la nube con servicios de analytics
        quicksight = Custom("Amazon QuickSight", "./icons/quicksight.png")  # # Dashboard interactivo para análisis avanzado de resultados
        export_cloud = Python("Exportacion")
        # # Función Lambda para generar y enviar reportes PDF por email o descarga

    # Flujo en nube
    usuario_cloud >> Edge(label="Acceso web") >> streamlit_cloud
    streamlit_cloud >> Edge(label="Audio/Texto") >> lambda_preprocessing
    lambda_preprocessing >> Edge(label="Datos procesados") >> sagemaker_model
    sagemaker_model >> Edge(label="Predicciones") >> quicksight
    quicksight >> Edge(label="Visualizaciones") >> streamlit_cloud
    sagemaker_model >> Edge(label="Resultados") >> s3_storage
    lambda_preprocessing >> Edge(label="Logs") >> cloudwatch_logs
    export_cloud >> Edge(label="Reportes") >> s3_storage
    streamlit_cloud >> Edge(label="Resultados") >> usuario_cloud

# ============================================================================
# EJECUCIÓN DEL DIAGRAMA
# ============================================================================

if __name__ == "__main__":
    import os
    # Agregar Graphviz al PATH para esta sesión
    os.environ['PATH'] = r'C:\Program Files\Graphviz\bin;' + os.environ.get('PATH', '')

    # Configurar graphviz para usar el ejecutable correcto
    import graphviz
    graphviz.set_jupyter_format('png')

    # Generar diagrama local
    diag_local
    print("Diagrama local generado: arquitectura_demenciascan_local.png")

    # Generar diagrama cloud (opcional)
    diag_cloud
    print("Diagrama cloud generado: arquitectura_demenciascan_cloud.png")
