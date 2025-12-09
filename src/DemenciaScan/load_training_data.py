"""
Script para cargar y procesar datos de entrenamiento desde la carpeta TranscripcionesParaEntrenarLLM
Preserva errores gramaticales y patrones de habla como indicadores de demencia
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from langdetect import detect, DetectorFactory
import warnings

warnings.filterwarnings('ignore')
DetectorFactory.seed = 0

class TranscriptionLoader:
    """Cargador de transcripciones para entrenamiento del modelo"""
    
    def __init__(self, transcriptions_folder="TranscripcionesParaEntrenarLLM/"):
        self.transcriptions_folder = transcriptions_folder
        self.data = []
        
    def extract_patient_info_from_filename(self, filename):
        """Extrae información del paciente desde el nombre del archivo"""
        # Formato: adrso###.txt
        # adrso podría indicar "Alzheimer Disease Research Study Oral"
        match = re.match(r'adrso(\d+)\.txt', filename)
        if match:
            patient_id = int(match.group(1))
            return {
                'patient_id': patient_id,
                'study_type': 'adrso',  # Alzheimer Disease Research Study Oral
                'filename': filename
            }
        return None
    
    def detect_language_content(self, text):
        """Detecta el idioma predominante del texto SIN modificarlo"""
        if not text or len(text.strip()) < 10:
            return 'unknown'
        
        try:
            return detect(text)
        except:
            # Fallback: detectar por palabras clave
            english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you']
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por']
            
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if word in text_lower)
            spanish_count = sum(1 for word in spanish_words if word in text_lower)
            
            if english_count > spanish_count:
                return 'en'
            elif spanish_count > english_count:
                return 'es'
            else:
                return 'unknown'
    
    def analyze_dementia_indicators(self, text):
        """
        Analiza indicadores específicos de demencia en el habla
        PRESERVANDO todos los errores y patrones originales
        """
        if not text:
            return {
                'audio_comprehension_issues': 0,
                'repetitive_patterns': 0,
                'incomplete_thoughts': 0,
                'filler_words_count': 0,
                'language_switching': 0,
                'semantic_errors': 0,
                'phonemic_errors': 0,
                'word_finding_pauses': 0
            }
        
        # 1. Problemas de comprensión auditiva
        audio_issues = len(re.findall(r'\[No se pudo entender el audio\]', text))
        
        # 2. Patrones repetitivos (palabras o frases repetidas)
        words = text.lower().split()
        repetitive_patterns = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                repetitive_patterns += 1
        
        # 3. Pensamientos incompletos (oraciones que se cortan abruptamente)
        sentences = text.split('.')
        incomplete_thoughts = 0
        for sentence in sentences[:-1]:  # Excluir la última oración
            if re.search(r'\b\w+\s*$', sentence.strip()):
                incomplete_thoughts += 1
        
        # 4. Palabras de relleno y muletillas
        filler_patterns = [
            r'\bah+\b', r'\beh+\b', r'\bum+\b', r'\bmm+\b', r'\buhm+\b',
            r'\bokay\b', r'\bueno\b', r'\bpues\b', r'\bentonces\b',
            r'\bjajaja\b', r'\bhola\b(?=\s+\bhola\b)',  # Repetición de hola
            r'\bgracias\b(?=.*\bgracias\b)',  # Repetición de gracias
        ]
        filler_count = sum(len(re.findall(pattern, text.lower())) for pattern in filler_patterns)
        
        # 5. Cambio de idioma (code-switching) - indicador de confusión
        # Contar transiciones entre inglés y español
        sentences = re.split(r'[.!?]+', text)
        language_switches = 0
        prev_lang = None
        for sentence in sentences:
            if len(sentence.strip()) > 5:
                try:
                    curr_lang = detect(sentence)
                    if prev_lang and prev_lang != curr_lang and curr_lang in ['en', 'es']:
                        language_switches += 1
                    prev_lang = curr_lang
                except:
                    pass
        
        # 6. Errores semánticos (palabras fuera de contexto)
        # Buscar patrones que sugieren confusión semántica
        semantic_error_patterns = [
            r'\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\b',  # Secuencias largas sin puntuación
            r'\b(?:spider-man|facebook|youtube|whatsapp)\b',  # Palabras fuera de contexto en descripción de imagen
        ]
        semantic_errors = sum(len(re.findall(pattern, text.lower())) for pattern in semantic_error_patterns)
        
        # 7. Errores fonémicos (aproximaciones de palabras)
        phonemic_patterns = [
            r'\b\w{1,2}\b(?!\s+(?:is|el|la|de|to|in|on))',  # Palabras muy cortas que no son artículos/preposiciones
            r'\b\w+ing\b(?=\s+\w+ing\b)',  # Repetición de terminaciones
        ]
        phonemic_errors = sum(len(re.findall(pattern, text.lower())) for pattern in phonemic_patterns)
        
        # 8. Pausas para encontrar palabras (indicadas por puntos suspensivos o espacios largos)
        word_finding_pauses = len(re.findall(r'\.{2,}|\s{3,}', text))
        
        return {
            'audio_comprehension_issues': audio_issues,
            'repetitive_patterns': repetitive_patterns,
            'incomplete_thoughts': incomplete_thoughts,
            'filler_words_count': filler_count,
            'language_switching': language_switches,
            'semantic_errors': semantic_errors,
            'phonemic_errors': phonemic_errors,
            'word_finding_pauses': word_finding_pauses
        }
    
    def calculate_discourse_metrics(self, text):
        """Calcula métricas del discurso preservando errores"""
        if not text:
            return {
                'total_utterances': 0,
                'avg_utterance_length': 0,
                'coherence_breaks': 0,
                'topic_maintenance': 0
            }
        
        # Dividir en enunciados (por puntuación o pausas largas)
        utterances = re.split(r'[.!?]+|\n+', text)
        valid_utterances = [u.strip() for u in utterances if len(u.strip()) > 3]
        
        total_utterances = len(valid_utterances)
        avg_utterance_length = np.mean([len(u.split()) for u in valid_utterances]) if valid_utterances else 0
        
        # Rupturas de coherencia (cambios abruptos de tema)
        coherence_breaks = len(re.findall(r'\b(?:okay|bueno|entonces)\b.*\b(?:okay|bueno|entonces)\b', text.lower()))
        
        # Mantenimiento del tema (menciones relacionadas con la tarea de descripción)
        task_related_words = ['picture', 'imagen', 'see', 'ver', 'happening', 'pasando', 'water', 'agua', 'cookies', 'galletas']
        topic_mentions = sum(1 for word in task_related_words if word in text.lower())
        topic_maintenance = topic_mentions / len(text.split()) if len(text.split()) > 0 else 0
        
        return {
            'total_utterances': total_utterances,
            'avg_utterance_length': avg_utterance_length,
            'coherence_breaks': coherence_breaks,
            'topic_maintenance': topic_maintenance
        }
    
    def load_transcriptions(self):
        """Carga todas las transcripciones PRESERVANDO errores y patrones originales"""
        if not os.path.exists(self.transcriptions_folder):
            raise FileNotFoundError(f"No se encontró la carpeta: {self.transcriptions_folder}")
        
        print(f"🔄 Cargando transcripciones desde: {self.transcriptions_folder}")
        print("📝 PRESERVANDO errores gramaticales y patrones de habla como indicadores de demencia")
        
        for filename in os.listdir(self.transcriptions_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.transcriptions_folder, filename)
                
                # Extraer información del paciente
                patient_info = self.extract_patient_info_from_filename(filename)
                if not patient_info:
                    continue
                
                # Leer contenido del archivo SIN LIMPIAR
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        original_text = f.read()
                except:
                    print(f"⚠️  Error leyendo archivo: {filename}")
                    continue
                
                if len(original_text.strip()) < 5:  # Saltar archivos completamente vacíos
                    continue
                
                # Detectar idioma SIN modificar el texto
                language = self.detect_language_content(original_text)
                
                # Analizar indicadores de demencia
                dementia_indicators = self.analyze_dementia_indicators(original_text)
                
                # Calcular métricas del discurso
                discourse_metrics = self.calculate_discourse_metrics(original_text)
                
                # Calcular métricas básicas PRESERVANDO errores
                words = original_text.split()
                unique_words = set(word.lower() for word in words if len(word) > 0)
                
                # Crear registro PRESERVANDO el texto original
                record = {
                    'id': patient_info['filename'].replace('.txt', ''),
                    'patient_id': patient_info['patient_id'],
                    'filename': filename,
                    'texto': original_text,  # TEXTO ORIGINAL SIN LIMPIAR
                    'texto_limpio': original_text,  # Mismo texto para compatibilidad con pipeline existente
                    'idioma': language,
                    'num_palabras': len(words),
                    'riqueza_lexica': len(unique_words) / len(words) if len(words) > 0 else 0,
                    
                    # Indicadores específicos de demencia
                    'audio_comprehension_issues': dementia_indicators['audio_comprehension_issues'],
                    'repetitive_patterns': dementia_indicators['repetitive_patterns'],
                    'incomplete_thoughts': dementia_indicators['incomplete_thoughts'],
                    'filler_words_count': dementia_indicators['filler_words_count'],
                    'language_switching': dementia_indicators['language_switching'],
                    'semantic_errors': dementia_indicators['semantic_errors'],
                    'phonemic_errors': dementia_indicators['phonemic_errors'],
                    'word_finding_pauses': dementia_indicators['word_finding_pauses'],
                    
                    # Métricas del discurso
                    'total_utterances': discourse_metrics['total_utterances'],
                    'avg_utterance_length': discourse_metrics['avg_utterance_length'],
                    'coherence_breaks': discourse_metrics['coherence_breaks'],
                    'topic_maintenance': discourse_metrics['topic_maintenance'],
                    
                    # Campos que necesitarán ser completados
                    'edad': None,
                    'nivel_educacion': None,
                    'calidad_sueno': None,
                    'habitos_diarios': None,
                    'nivel_entorno_social': None,
                    'diagnostico_alzheimer': None  # Campo objetivo
                }
                
                self.data.append(record)
        
        print(f"✅ Cargadas {len(self.data)} transcripciones CON errores preservados")
        return self.data
    
    def create_dataframe(self):
        """Crea un DataFrame con los datos cargados"""
        if not self.data:
            self.load_transcriptions()
        
        df = pd.DataFrame(self.data)
        return df
    
    def estimate_dementia_labels(self, df):
        """
        Estima etiquetas de demencia basándose en indicadores del habla
        NOTA: En un caso real, usar diagnósticos clínicos reales.
        """
        print("⚠️  ADVERTENCIA: Estimando etiquetas basándose en indicadores del habla.")
        print("   En un entorno real, usa diagnósticos clínicos reales.")
        
        # Crear score compuesto basado en indicadores de deterioro cognitivo
        df['dementia_risk_score'] = (
            df['audio_comprehension_issues'] * 0.15 +
            df['repetitive_patterns'] * 0.20 +
            df['incomplete_thoughts'] * 0.15 +
            df['filler_words_count'] * 0.10 +
            df['language_switching'] * 0.10 +
            df['semantic_errors'] * 0.15 +
            df['phonemic_errors'] * 0.10 +
            df['word_finding_pauses'] * 0.05
        )
        
        # Normalizar el score
        max_score = df['dementia_risk_score'].max()
        if max_score > 0:
            df['dementia_risk_score'] = df['dementia_risk_score'] / max_score
        
        # Asignar etiquetas basándose en el score
        # Usar percentil 60 como umbral (más conservador)
        threshold = df['dementia_risk_score'].quantile(0.6)
        df['diagnostico_alzheimer'] = (df['dementia_risk_score'] > threshold).map({True: 'si', False: 'no'})
        
        # Estimar campos demográficos (para demostración)
        np.random.seed(42)
        df['edad'] = np.random.randint(65, 85, len(df))  # Rango típico para estudios de demencia
        df['nivel_educacion'] = np.random.choice(['primaria', 'secundaria', 'universitaria'], len(df), p=[0.3, 0.4, 0.3])
        df['calidad_sueno'] = np.random.choice(['buena', 'regular', 'mala'], len(df), p=[0.3, 0.4, 0.3])
        df['habitos_diarios'] = np.random.choice(['sedentario', 'ejercicio regular'], len(df), p=[0.6, 0.4])
        df['nivel_entorno_social'] = np.random.choice(['bajo', 'medio', 'alto'], len(df), p=[0.3, 0.4, 0.3])
        
        return df
    
    def save_processed_data(self, df, output_path="transcripciones_entrenamiento_procesadas.csv"):
        """Guarda los datos procesados"""
        df.to_csv(output_path, index=False)
        print(f"💾 Datos guardados en: {output_path}")
        
        # Mostrar estadísticas
        print(f"\n📊 ESTADÍSTICAS DEL DATASET (CON ERRORES PRESERVADOS):")
        print(f"   Total de muestras: {len(df)}")
        print(f"   Idiomas detectados:")
        print(df['idioma'].value_counts().to_string())
        print(f"\n   Distribución de diagnósticos estimados:")
        print(df['diagnostico_alzheimer'].value_counts().to_string())
        print(f"\n   Indicadores de demencia promedio:")
        print(f"   - Problemas de comprensión auditiva: {df['audio_comprehension_issues'].mean():.1f}")
        print(f"   - Patrones repetitivos: {df['repetitive_patterns'].mean():.1f}")
        print(f"   - Pensamientos incompletos: {df['incomplete_thoughts'].mean():.1f}")
        print(f"   - Palabras de relleno: {df['filler_words_count'].mean():.1f}")
        print(f"   - Cambios de idioma: {df['language_switching'].mean():.1f}")
        print(f"   - Score de riesgo de demencia: {df['dementia_risk_score'].mean():.3f}")
        
        return output_path

def main():
    """Función principal para procesar las transcripciones"""
    print("🚀 Procesando transcripciones para entrenamiento del modelo BERT...")
    print("🧠 PRESERVANDO errores gramaticales como indicadores de demencia")
    
    try:
        # Crear cargador
        loader = TranscriptionLoader()
        
        # Cargar y procesar datos
        df = loader.create_dataframe()
        
        if len(df) == 0:
            print("❌ No se encontraron transcripciones válidas")
            return
        
        # Estimar etiquetas basándose en indicadores del habla
        df = loader.estimate_dementia_labels(df)
        
        # Guardar datos procesados
        output_path = loader.save_processed_data(df)
        
        print(f"\n✅ Procesamiento completado!")
        print(f"📁 Archivo generado: {output_path}")
        print(f"🔄 Ahora puedes usar este archivo con el pipeline BERT existente")
        print(f"🧠 Los errores gramaticales y patrones de habla se preservaron como características")
        
        # Mostrar algunas muestras
        print(f"\n📋 MUESTRA DE DATOS:")
        sample_cols = ['id', 'idioma', 'num_palabras', 'dementia_risk_score', 'diagnostico_alzheimer']
        print(df[sample_cols].head(10).to_string(index=False))
        
        # Mostrar ejemplo de texto preservado
        print(f"\n📝 EJEMPLO DE TEXTO PRESERVADO (con errores):")
        sample_text = df.iloc[0]['texto'][:200] + "..."
        print(f"   {sample_text}")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
