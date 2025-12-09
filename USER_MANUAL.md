# Manual de Usuario - DemenciaScan

## Introducción
DemenciaScan es una herramienta de apoyo clínico diseñada para detectar posibles signos de deterioro cognitivo mediante el análisis de transcripciones de audio utilizando Inteligencia Artificial. Este manual detalla paso a paso cómo instalar, configurar y utilizar la aplicación.

## 1. Requisitos Previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu computadora:

1.  **Python (versión 3.8 o superior)**:
    *   Descárgalo desde [python.org](https://www.python.org/downloads/).
    *   Al instalar, asegúrate de marcar la casilla **"Add Python to PATH"**.
2.  **Git**:
    *   Descárgalo desde [git-scm.com](https://git-scm.com/downloads).
3.  **Visual Studio Code (Opcional pero recomendado)**:
    *   Un editor de código para facilitar la visualización de archivos. Descárgalo desde [code.visualstudio.com](https://code.visualstudio.com/).
4.  **Cuenta en Groq (para transcripción)**:
    *   Regístrate en [console.groq.com](https://console.groq.com/) y genera una **API Key**.

## 2. Instalación Paso a Paso

Sigue estos pasos para instalar la aplicación en tu sistema.

### Paso 2.1: Obtener el Código
Abre una terminal (Símbolo del sistema, PowerShell o Terminal de VS Code) y ejecuta:

```bash
git clone <url-del-repositorio>
cd DemenciaScan
```

### Paso 2.2: Crear un Entorno Virtual
Es altamente recomendable usar un entorno virtual para no mezclar las librerías con otras instalaciones de Python.

**En Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*Nota: Verás que aparece `(venv)` al inicio de la línea de comandos, indicando que el entorno está activo.*

### Paso 2.3: Instalar Dependencias
Instala las librerías necesarias para que la aplicación funcione:

```bash
pip install -r requirements.txt
```
*Este proceso puede tardar unos minutos dependiendo de tu conexión a internet.*

### Paso 2.4: Configuración de Claves (API Key)
1.  En la carpeta principal del proyecto, crea un nuevo archivo llamado `.env`.
2.  Abre este archivo con un editor de texto (Notepad o VS Code).
3.  Escribe lo siguiente, reemplazando `tu_api_key_aqui` por la clave que obtuviste de Groq:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
4.  Guarda y cierra el archivo.

### Paso 2.5: Configuración del Modelo IA
Para asegurarte de que el modelo de inteligencia artificial esté listo, ejecuta el script de configuración:

```bash
python scripts/setup_bert.py
```
*Si todo está correcto, verás mensajes en verde indicando "OK".*

## 3. Ejecución de la Aplicación

Una vez instalado todo, inicia la aplicación con el siguiente comando:

```bash
streamlit run src/app_streamlit.py
```

*   La aplicación se abrirá automáticamente en tu navegador web predeterminado.
*   Si no se abre, copia la dirección que aparece en la terminal (usualmente `http://localhost:8501`) y pégala en tu navegador.

## 4. Guía de Uso Detallada

### 4.1. Pantalla de Inicio
Al entrar, verás el panel de control principal.
*   **Barra lateral izquierda**: Menú de navegación para cambiar entre las diferentes funciones.
*   **Panel central**: Información general y estado del sistema (verificará si el modelo y la API están conectados).

### 4.2. Módulo de Transcripción (Grabar Audio)
Si deseas analizar una conversación en vivo:
1.  Ve a la sección **"🎤 Transcripción"** en el menú lateral.
2.  Selecciona el método de entrada (Micrófono).
3.  Presiona el botón **"Iniciar Grabación"**.
4.  Habla claramente al micrófono.
5.  Presiona **"Detener"** cuando termines.
6.  El texto aparecerá en pantalla. Puedes copiarlo o enviarlo directamente a análisis.

### 4.3. Módulo de Análisis de Texto
Si ya tienes un texto o acabas de transcribir uno:
1.  Ve a **"📝 Análisis de Texto"**.
2.  **Ingresar Texto**:
    *   Opción A: Pega el texto en el cuadro.
    *   Opción B: Sube un archivo `.txt`.
    *   Opción C: Usa el botón "Cargar Ejemplo" para probar el sistema.
3.  **Datos del Paciente (Importante)**:
    *   Rellena los campos: Edad, Nivel Educativo, Calidad de Sueño, etc.
    *   *Estos datos ayudan a la IA a ser más precisa, ya que el lenguaje cambia con la edad y educación.*
4.  Haz clic en el botón **"🔍 Analizar Texto"**.

### 4.4. Interpretación de Resultados
Después de unos segundos, verás el reporte:

*   **Probabilidad de Riesgo**: Un medidor de 0 a 100%.
    *   🟢 **Verde (<50%)**: Bajo riesgo.
    *   🟡 **Amarillo (50-70%)**: Riesgo moderado.
    *   🔴 **Rojo (>70%)**: Alto riesgo.
*   **Indicadores Detectados**: Lista de características lingüísticas encontradas (ej. "Pausas prolongadas", "Repetición de palabras").
*   **Gráficos**:
    *   Comparativa con población normal.
    *   Radar de características lingüísticas.

### 4.5. Exportar Informe
Al final de la página de resultados encontrarás botones para descargar el reporte:
*   **PDF**: Para imprimir o guardar en historia clínica.
*   **CSV/JSON**: Para análisis de datos o investigación.

## 5. Solución de Problemas Frecuentes

| Problema | Posible Causa | Solución |
| :--- | :--- | :--- |
| **"Command not found: python"** | Python no está en el PATH | Reinstala Python y marca "Add to PATH". O usa `python3` en lugar de `python`. |
| **"ModuleNotFoundError"** | Dependencias no instaladas | Ejecuta `pip install -r requirements.txt` con el entorno virtual activado. |
| **Error de API Groq** | Clave incorrecta o faltante | Verifica que el archivo `.env` tenga la clave correcta sin espacios extra. |
| **El navegador no abre** | Bloqueo de pop-ups | Abre manualmente `http://localhost:8501` en Chrome o Firefox. |
| **Micrófono no funciona** | Permisos del navegador | Dale permiso al navegador para acceder al micrófono cuando te lo pida. |

## 6. Soporte y Contacto
Si encuentras un error no listado aquí:
1.  Toma una captura de pantalla del error en la terminal.
2.  Copia el mensaje de error.
3.  Contacta al equipo de desarrollo o abre un "Issue" en el repositorio de GitHub.
