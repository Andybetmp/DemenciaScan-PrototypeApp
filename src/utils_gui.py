"""
Utilidades para la interfaz gráfica de DemenciaScan
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional

from config import (
    CLASSIFICATION_THRESHOLDS,
    THEME_COLORS,
    PLOT_CONFIG,
    EXPORTS_DIR,
    MESSAGES
)


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def create_probability_gauge(probability: float, title: str = "Probabilidad") -> go.Figure:
    """
    Crea un gráfico de gauge para mostrar probabilidades
    
    Args:
        probability: Valor de probabilidad (0-1)
        title: Título del gráfico
    
    Returns:
        Figura de Plotly
    """
    # Determinar color según umbral
    if probability >= CLASSIFICATION_THRESHOLDS["high_risk"]:
        color = THEME_COLORS["danger"]
    elif probability >= CLASSIFICATION_THRESHOLDS["moderate_risk"]:
        color = THEME_COLORS["warning"]
    else:
        color = THEME_COLORS["success"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig


def create_comparison_bars(prob_no_risk: float, prob_risk: float) -> go.Figure:
    """
    Crea gráfico de barras comparativo de probabilidades
    
    Args:
        prob_no_risk: Probabilidad sin riesgo
        prob_risk: Probabilidad con riesgo
    
    Returns:
        Figura de Plotly
    """
    fig = go.Figure(data=[
        go.Bar(
            name='Probabilidades',
            x=['Sin Riesgo', 'Con Riesgo'],
            y=[prob_no_risk * 100, prob_risk * 100],
            marker_color=[THEME_COLORS["success"], THEME_COLORS["danger"]],
            text=[f'{prob_no_risk*100:.1f}%', f'{prob_risk*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Comparación de Probabilidades",
        yaxis_title="Probabilidad (%)",
        xaxis_title="Clasificación",
        height=400,
        showlegend=False,
        template=PLOT_CONFIG["template"]
    )
    
    return fig


def create_indicators_radar(indicators: Dict[str, int]) -> go.Figure:
    """
    Crea gráfico de radar para indicadores de demencia
    
    Args:
        indicators: Diccionario con indicadores y sus valores
    
    Returns:
        Figura de Plotly
    """
    categories = list(indicators.keys())
    values = list(indicators.values())
    
    # Normalizar valores a escala 0-10
    max_val = max(values) if max(values) > 0 else 1
    normalized_values = [(v / max_val) * 10 for v in values]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        fillcolor=THEME_COLORS["warning"],
        opacity=0.6,
        line=dict(color=THEME_COLORS["danger"], width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False,
        title="Indicadores de Demencia Detectados",
        height=500
    )
    
    return fig


def create_feature_importance_chart(features: Dict[str, float]) -> go.Figure:
    """
    Crea gráfico de importancia de características
    
    Args:
        features: Diccionario con características y sus valores
    
    Returns:
        Figura de Plotly
    """
    df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
    df = df.sort_values('Value', ascending=True)
    
    fig = px.bar(
        df,
        x='Value',
        y='Feature',
        orientation='h',
        title="Características Analizadas",
        color='Value',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template=PLOT_CONFIG["template"]
    )
    
    return fig


def create_timeline_chart(history: List[Dict]) -> go.Figure:
    """
    Crea gráfico de línea temporal de análisis
    
    Args:
        history: Lista de análisis previos
    
    Returns:
        Figura de Plotly
    """
    if not history:
        return go.Figure()
    
    df = pd.DataFrame(history)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['fecha'],
        y=df['probabilidad_con_riesgo'] * 100,
        mode='lines+markers',
        name='Riesgo de Demencia',
        line=dict(color=THEME_COLORS["danger"], width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Evolución Temporal de Análisis",
        xaxis_title="Fecha",
        yaxis_title="Probabilidad de Riesgo (%)",
        height=400,
        template=PLOT_CONFIG["template"]
    )
    
    return fig


# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def export_to_json(data: Dict, filename: str = None) -> str:
    """
    Exporta resultados a JSON
    
    Args:
        data: Datos a exportar
        filename: Nombre del archivo (opcional)
    
    Returns:
        Ruta del archivo creado
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_{timestamp}.json"
    
    filepath = EXPORTS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def export_to_csv(data: Dict, filename: str = None) -> str:
    """
    Exporta resultados a CSV
    
    Args:
        data: Datos a exportar
        filename: Nombre del archivo (opcional)
    
    Returns:
        Ruta del archivo creado
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_{timestamp}.csv"
    
    filepath = EXPORTS_DIR / filename
    
    # Convertir a DataFrame
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    return str(filepath)


def generate_pdf_report(data: Dict, filename: str = None) -> str:
    """
    Genera reporte en PDF con diseño profesional
    
    Args:
        data: Datos del análisis
        filename: Nombre del archivo (opcional)
    
    Returns:
        Ruta del archivo creado
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf no está instalado. Ejecuta: pip install fpdf")
    
    def sanitize_text(text):
        """Elimina caracteres no soportados por latin-1 (emojis, etc)"""
        if not isinstance(text, str):
            return str(text)
        return text.encode('latin-1', 'replace').decode('latin-1').replace('?', '')

    class PDF(FPDF):
        def header(self):
            # Fondo del encabezado
            self.set_fill_color(31, 119, 180)  # Azul profesional
            self.rect(0, 0, 210, 40, 'F')
            
            # Título
            self.set_font('Arial', 'B', 24)
            self.set_text_color(255, 255, 255)
            self.set_y(10)
            self.cell(0, 10, 'DemenciaScan', 0, 1, 'C')
            
            # Subtítulo
            self.set_font('Arial', '', 12)
            self.cell(0, 10, sanitize_text('Reporte de Análisis Cognitivo con IA'), 0, 1, 'C')
            self.ln(20)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, sanitize_text(f'Página {self.page_no()} | Generado por DemenciaScan | {datetime.now().strftime("%Y-%m-%d")}'), 0, 0, 'C')

        def chapter_title(self, label):
            self.set_font('Arial', 'B', 14)
            self.set_text_color(31, 119, 180)
            self.cell(0, 10, sanitize_text(label), 0, 1, 'L')
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def chapter_body(self, body):
            self.set_font('Arial', '', 11)
            self.set_text_color(0)
            self.multi_cell(0, 6, sanitize_text(body))
            self.ln()

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reporte_{timestamp}.pdf"
    
    filepath = EXPORTS_DIR / filename
    
    # Crear PDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 1. Información General
    pdf.chapter_title("1. Información General")
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 8, sanitize_text("Fecha de Análisis:"), 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 1)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 8, "ID Archivo:", 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, sanitize_text(data.get('archivo', 'N/A')), 0, 1)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 8, "Idioma Detectado:", 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, sanitize_text(data.get('idioma_detectado', 'N/A')), 0, 1)
    pdf.ln(5)
    
    # 2. Resultados del Análisis
    pdf.chapter_title("2. Resultados del Diagnóstico")
    
    # Cuadro de resultado principal
    prob_risk = data.get('probabilidad_con_riesgo', 0)
    
    if prob_risk >= CLASSIFICATION_THRESHOLDS["high_risk"]:
        bg_color = (255, 235, 238)  # Rojo claro
        text_color = (183, 28, 28)  # Rojo oscuro
        status = "ALTO RIESGO"
    elif prob_risk >= CLASSIFICATION_THRESHOLDS["moderate_risk"]:
        bg_color = (255, 243, 224)  # Naranja claro
        text_color = (230, 81, 0)   # Naranja oscuro
        status = "RIESGO MODERADO"
    else:
        bg_color = (232, 245, 233)  # Verde claro
        text_color = (27, 94, 32)   # Verde oscuro
        status = "BAJO RIESGO"
        
    pdf.set_fill_color(*bg_color)
    pdf.set_text_color(*text_color)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, sanitize_text(f"{status} ({prob_risk*100:.1f}%)"), 0, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(50, 8, "Predicción del Modelo:", 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, sanitize_text(data.get('prediccion', 'N/A')), 0, 1)
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(50, 8, "Nivel de Confianza:", 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"{data.get('confianza', 0)*100:.1f}%", 0, 1)
    pdf.ln(5)
    
    # 3. Probabilidades Detalladas
    pdf.chapter_title("3. Desglose de Probabilidades")
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(95, 10, sanitize_text("Categoría"), 1, 0, 'C', 1)
    pdf.cell(95, 10, "Probabilidad", 1, 1, 'C', 1)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(95, 10, "Sin Riesgo de Demencia", 1, 0, 'L')
    pdf.cell(95, 10, f"{data.get('probabilidad_sin_riesgo', 0)*100:.1f}%", 1, 1, 'C')
    
    pdf.cell(95, 10, "Con Riesgo de Demencia", 1, 0, 'L')
    pdf.cell(95, 10, f"{data.get('probabilidad_con_riesgo', 0)*100:.1f}%", 1, 1, 'C')
    pdf.ln(10)
    
    # 4. Indicadores Clave
    pdf.chapter_title("4. Indicadores Lingüísticos Detectados")
    
    if 'indicadores_clave' in data:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(140, 10, "Indicador", 1, 0, 'C', 1)
        pdf.cell(50, 10, "Valor Detectado", 1, 1, 'C', 1)
        
        pdf.set_font('Arial', '', 11)
        for key, value in data['indicadores_clave'].items():
            readable_key = key.replace('_', ' ').title()
            pdf.cell(140, 8, sanitize_text(readable_key), 1)
            
            # Colorear valor si es alto
            if value > 5:
                pdf.set_text_color(200, 0, 0)
                pdf.set_font('Arial', 'B', 11)
            elif value > 2:
                pdf.set_text_color(200, 100, 0)
            else:
                pdf.set_text_color(0, 100, 0)
                
            pdf.cell(50, 8, str(value), 1, 1, 'C')
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 11)
    else:
        pdf.cell(0, 10, sanitize_text("No hay indicadores detallados disponibles."), 0, 1)
    
    pdf.ln(10)
    
    # 5. Interpretación y Recomendaciones
    pdf.chapter_title("5. Interpretación")
    
    interpretation = get_interpretation_message(prob_risk)
    pdf.set_font('Arial', 'I', 11)
    pdf.multi_cell(0, 6, sanitize_text(interpretation))
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_draw_color(200, 0, 0)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(100)
    pdf.cell(0, 5, "AVISO LEGAL IMPORTANTE:", 0, 1)
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, sanitize_text(MESSAGES["disclaimer"]))
    
    # Guardar
    pdf.output(str(filepath))
    
    return str(filepath)


# ============================================================================
# FUNCIONES DE FORMATO Y PRESENTACIÓN
# ============================================================================

def format_prediction_result(results: Dict) -> str:
    """
    Formatea los resultados de predicción para mostrar
    
    Args:
        results: Diccionario con resultados
    
    Returns:
        String formateado con HTML
    """
    prob_risk = results.get('probabilidad_con_riesgo', 0)
    
    # Determinar nivel de riesgo
    if prob_risk >= CLASSIFICATION_THRESHOLDS["high_risk"]:
        risk_level = "ALTO RIESGO"
        color = THEME_COLORS["danger"]
        icon = "⚠️"
    elif prob_risk >= CLASSIFICATION_THRESHOLDS["moderate_risk"]:
        risk_level = "RIESGO MODERADO"
        color = THEME_COLORS["warning"]
        icon = "⚡"
    else:
        risk_level = "BAJO RIESGO"
        color = THEME_COLORS["success"]
        icon = "✅"
    
    html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color};">
        <h2 style="color: {color}; margin: 0;">{icon} {risk_level}</h2>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>Predicción:</strong> {results.get('prediccion', 'N/A')}
        </p>
        <p style="font-size: 16px; margin: 5px 0;">
            <strong>Confianza:</strong> {results.get('confianza', 0)*100:.1f}%
        </p>
    </div>
    """
    
    return html


def format_indicators_table(indicators: Dict[str, int]) -> pd.DataFrame:
    """
    Formatea indicadores como tabla
    
    Args:
        indicators: Diccionario con indicadores
    
    Returns:
        DataFrame formateado
    """
    # Nombres legibles
    readable_names = {
        'problemas_audio': 'Problemas de Comprensión Auditiva',
        'patrones_repetitivos': 'Patrones Repetitivos',
        'muletillas': 'Muletillas y Pausas',
        'errores_semanticos': 'Errores Semánticos',
        'errores_fonémicos': 'Errores Fonémicos'
    }
    
    data = []
    for key, value in indicators.items():
        readable_key = readable_names.get(key, key.replace('_', ' ').title())
        
        # Determinar estado
        if value > 5:
            estado = "🔴 Alto"
            color = THEME_COLORS["danger"]
        elif value > 2:
            estado = "🟡 Moderado"
            color = THEME_COLORS["warning"]
        else:
            estado = "🟢 Bajo"
            color = THEME_COLORS["success"]
        
        data.append({
            'Indicador': readable_key,
            'Valor': value,
            'Estado': estado
        })
    
    return pd.DataFrame(data)


def get_interpretation_message(probability: float) -> str:
    """
    Obtiene mensaje de interpretación según probabilidad
    
    Args:
        probability: Probabilidad de riesgo
    
    Returns:
        Mensaje de interpretación
    """
    if probability >= CLASSIFICATION_THRESHOLDS["high_risk"]:
        return MESSAGES["high_risk"]
    elif probability >= CLASSIFICATION_THRESHOLDS["moderate_risk"]:
        return MESSAGES["moderate_risk"]
    else:
        return MESSAGES["low_risk"]


# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Valida entrada de texto
    
    Args:
        text: Texto a validar
    
    Returns:
        Tupla (es_válido, mensaje)
    """
    if not text or len(text.strip()) == 0:
        return False, "El texto está vacío"
    
    if len(text.strip()) < 10:
        return False, "El texto es demasiado corto (mínimo 10 caracteres)"
    
    words = text.split()
    if len(words) < 5:
        return False, "El texto debe contener al menos 5 palabras"
    
    return True, "Texto válido"


def validate_demographic_data(data: Dict) -> Tuple[bool, str]:
    """
    Valida datos demográficos
    
    Args:
        data: Diccionario con datos demográficos
    
    Returns:
        Tupla (es_válido, mensaje)
    """
    required_fields = ['edad', 'nivel_educacion', 'calidad_sueno', 
                      'habitos_diarios', 'nivel_entorno_social']
    
    for field in required_fields:
        if field not in data:
            return False, f"Falta el campo: {field}"
    
    # Validar edad
    edad = data.get('edad', 0)
    if not isinstance(edad, (int, float)) or edad < 0 or edad > 120:
        return False, "Edad inválida (debe estar entre 0 y 120)"
    
    return True, "Datos válidos"


# ============================================================================
# FUNCIONES DE HISTORIAL
# ============================================================================

def save_to_history(results: Dict, history_file: str = "history.json") -> None:
    """
    Guarda análisis en historial
    
    Args:
        results: Resultados del análisis
        history_file: Archivo de historial
    """
    history_path = EXPORTS_DIR / history_file
    
    # Cargar historial existente
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    # Agregar timestamp
    results['fecha'] = datetime.now().isoformat()
    
    # Agregar a historial
    history.append(results)
    
    # Guardar
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_history(history_file: str = "history.json") -> List[Dict]:
    """
    Carga historial de análisis
    
    Args:
        history_file: Archivo de historial
    
    Returns:
        Lista de análisis previos
    """
    history_path = EXPORTS_DIR / history_file
    
    if not history_path.exists():
        return []
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history


def clear_history(history_file: str = "history.json") -> None:
    """
    Limpia el historial
    
    Args:
        history_file: Archivo de historial
    """
    history_path = EXPORTS_DIR / history_file
    
    if history_path.exists():
        history_path.unlink()


# ============================================================================
# FUNCIONES DE AYUDA
# ============================================================================

def get_example_transcription() -> str:
    """
    Retorna una transcripción de ejemplo
    
    Returns:
        Texto de ejemplo
    """
    return """Hola, estoy describiendo lo que veo en esta imagen. 
Veo una cocina donde hay una mujer lavando platos. 
También hay niños que están... eh... ¿cómo se dice? 
Tratando de alcanzar unas galletas. 
El agua se está derramando por el suelo. 
La mujer no se da cuenta de que el agua... el agua se está saliendo.
Los niños están parados en una silla que parece que se va a caer.
Es una situación peligrosa en la cocina."""


def format_file_size(size_bytes: int) -> str:
    """
    Formatea tamaño de archivo
    
    Args:
        size_bytes: Tamaño en bytes
    
    Returns:
        String formateado (ej: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    print("✅ Utilidades GUI cargadas correctamente")
    print(f"📁 Directorio de exportaciones: {EXPORTS_DIR}")
