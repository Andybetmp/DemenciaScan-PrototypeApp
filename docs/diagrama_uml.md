classDiagram
    class DemenciaScanApp {
        +main(): void
        +load_model(): Predictor
        +check_system_status(): dict
        +process_text_analysis(): dict
    }

    class TranscripcionPage {
        +grabar_audio(): tuple
        +guardar_audio(): str
        +transcribir_audio(): str
        +show_recording_interface(): void
    }

    class AnalisisTextoPage {
        +show_input_methods(): void
        +show_demographic_form(): void
        +display_results(): void
        +export_results(): void
    }

    class UtilsGUI {
        +create_probability_gauge(): Figure
        +create_comparison_bars(): Figure
        +create_indicators_radar(): Figure
        +format_prediction_result(): str
        +save_to_history(): void
        +load_history(): list
    }

    class DemenciaPredictor {
        +predict_single_file(): dict
        +preprocess_text(): str
        +extract_features(): dict
    }

    class Config {
        +STREAMLIT_CONFIG: dict
        +MODEL_PATH: Path
        +GROQ_API_KEY: str
        +CLASSIFICATION_THRESHOLDS: dict
        +THEME_COLORS: dict
    }

    class GroqClient {
        +audio.transcriptions.create(): str
    }

    DemenciaScanApp --> TranscripcionPage
    DemenciaScanApp --> AnalisisTextoPage
    DemenciaScanApp --> UtilsGUI
    DemenciaScanApp --> Config
    TranscripcionPage --> GroqClient
    AnalisisTextoPage --> DemenciaPredictor
    AnalisisTextoPage --> UtilsGUI
    UtilsGUI --> Config
    DemenciaPredictor --> Config
```

```mermaid
sequenceDiagram
    participant U as Usuario
    participant A as App Streamlit
    participant T as Transcripción
    participant G as Groq API
    participant P as Predictor
    participant H as Historial

    U->>A: Accede a página de Transcripción
    A->>T: Muestra interfaz de grabación
    U->>T: Presiona INS para grabar
    T->>T: Graba audio
    U->>T: Suelta INS para detener
    T->>T: Guarda audio temporal
    T->>G: Envía audio para transcripción
    G->>T: Retorna texto transcrito
    T->>A: Almacena transcripción en session_state
    T->>U: Muestra transcripción y copia al portapapeles

    U->>A: Navega a Análisis de Texto
    A->>P: Carga modelo BERT (si no está cargado)
    U->>A: Ingresa datos demográficos
    U->>A: Hace clic en "Analizar Texto"
    A->>P: Envía texto + datos demográficos
    P->>P: Preprocesa texto
    P->>P: Extrae características lingüísticas
    P->>P: Aplica modelo BERT
    P->>A: Retorna probabilidades y indicadores
    A->>H: Guarda resultados en historial
    A->>U: Muestra resultados con gráficos

    U->>A: Solicita exportar resultados
    A->>A: Genera archivo (JSON/CSV/PDF)
    A->>U: Descarga archivo
```

```mermaid
flowchart TD
    A[Usuario accede a DemenciaScan] --> B{¿Qué acción?}

    B -->|Transcribir Audio| C[Ir a página Transcripción]
    B -->|Analizar Texto| D[Ir a página Análisis de Texto]
    B -->|Ver Historial| E[Ir a página Historial]

    C --> F{¿Método de grabación?}
    F -->|Micrófono + Teclado| G[Mostrar instrucciones]
    F -->|Navegador| H[Mostrar interfaz WebRTC - Experimental]

    G --> I[Hacer clic 'Iniciar Grabación']
    I --> J[Presionar y mantener INS]
    J --> K[Grabar audio frames]
    J --> L[Soltar INS para detener]
    L --> M[Guardar audio como WAV temporal]
    M --> N[Enviar a Groq Whisper API]
    N --> O[Recibir transcripción]
    O --> P[Mostrar transcripción]
    P --> Q[Copiar al portapapeles]
    Q --> R[Almacenar en session_state]

    D --> S{¿Método de entrada?}
    S -->|Cargar archivo| T[Subir archivo .txt]
    S -->|Pegar texto| U[Pegar transcripción]
    S -->|Usar ejemplo| V[Cargar texto de ejemplo]

    T --> W[Leer contenido del archivo]
    U --> W
    V --> W

    W --> X[Mostrar formulario demográfico]
    X --> Y[Ingresar edad, educación, sueño, hábitos, entorno social]
    Y --> Z[Hacer clic 'Analizar Texto']

    Z --> AA{¿Modelo cargado?}
    AA -->|No| BB[Cargar modelo BERT]
    AA -->|Sí| CC[Continuar]

    BB --> CC[Crear archivo temporal con texto]
    CC --> DD[Enviar a DemenciaPredictor]
    DD --> EE[Preprocesar texto]
    EE --> FF[Extraer características lingüísticas]
    FF --> GG[Aplicar modelo BERT]
    GG --> HH[Calcular probabilidades]
    HH --> II[Generar indicadores de demencia]
    II --> JJ[Retornar resultados]

    JJ --> KK[Mostrar resultados principales]
    KK --> LL[Crear gráficos: Gauge + Barras + Radar]
    LL --> MM[Mostrar tabla de indicadores]
    MM --> NN[Mostrar interpretación]
    NN --> OO[Guardar en historial]

    OO --> PP{¿Exportar?}
    PP -->|JSON| QQ[Generar archivo JSON]
    PP -->|CSV| RR[Generar archivo CSV]
    PP -->|PDF| SS[Generar reporte PDF]
    PP -->|No| TT[Fin del análisis]

    QQ --> UU[Descargar archivo]
    RR --> UU
    SS --> UU
    UU --> TT

    E --> VV[Mostrar lista de análisis previos]
    VV --> WW{¿Más de 1 análisis?}
    WW -->|Sí| XX[Mostrar gráfico de evolución temporal]
    WW -->|No| YY[Mostrar solo tabla]

    XX --> ZZ[Permitir limpiar historial]
    YY --> ZZ
    ZZ --> TT
```

## 📋 Diagrama UML - Arquitectura de DemenciaScan

### 🏗️ Diagrama de Clases

El diagrama muestra las principales clases y sus relaciones:

1. **DemenciaScanApp**: Clase principal que maneja la aplicación Streamlit
2. **TranscripcionPage**: Maneja la funcionalidad de grabación y transcripción
3. **AnalisisTextoPage**: Gestiona el análisis de texto con el modelo BERT
4. **UtilsGUI**: Utilidades para visualización y manejo de datos
5. **DemenciaPredictor**: Clase que encapsula el modelo BERT y predicciones
6. **Config**: Configuraciones globales de la aplicación
7. **GroqClient**: Cliente para la API de transcripción de Groq

### 🔄 Diagrama de Secuencia

Muestra el flujo de interacción entre usuario, aplicación y servicios externos:

1. **Flujo de Transcripción**: Usuario → Grabación → API Groq → Resultado
2. **Flujo de Análisis**: Usuario → Datos → Modelo BERT → Resultados → Historial

### 📊 Diagrama de Flujo

Detalla el flujo completo de uso de la aplicación:

- **Rutas principales**: Transcripción, Análisis, Historial
- **Procesos internos**: Carga de modelo, preprocesamiento, predicción
- **Salidas**: Visualizaciones, exportaciones, almacenamiento

### 🔗 Relaciones Clave

- **Composición**: `DemenciaScanApp` contiene las páginas y utilidades
- **Dependencia**: Todas las clases dependen de `Config` para configuraciones
- **Uso**: `TranscripcionPage` usa `GroqClient`, `AnalisisTextoPage` usa `DemenciaPredictor`
- **Herencia/Implementación**: No hay herencia directa, pero composición fuerte

Este diagrama proporciona una visión clara de la arquitectura modular y el flujo de datos en la aplicación DemenciaScan.
