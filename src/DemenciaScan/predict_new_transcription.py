"""
Script para probar el modelo BERT entrenado con nuevas transcripciones
Permite hacer predicciones individuales sobre riesgo de demencia
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path

# Importar módulos locales
from data_preprocessing import DemenciaPreprocessor
from bert_model import MultimodalBERTClassifier
from load_training_data import TranscriptionLoader

class DemenciaPredictor:
    """Predictor para nuevas transcripciones usando el modelo entrenado"""
    
    def __init__(self, model_path="models_test/best_model.pth"):
        """
        Inicializa el predictor
        
        Args:
            model_path: Ruta al modelo entrenado
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo
        self.model, self.model_config, self.preprocessor = self._load_model()
        
        print(f"🤖 Modelo cargado desde: {model_path}")
        print(f"🖥️  Dispositivo: {self.device}")
    
    def _load_model(self):
        """Carga el modelo entrenado y configuración"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        # Crear modelo
        model = MultimodalBERTClassifier(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Crear preprocesador
        preprocessor = DemenciaPreprocessor(model_name=model_config['model_name'])
        
        return model, model_config, preprocessor
    
    def process_text_file(self, file_path):
        """
        Procesa un archivo de texto y extrae características
        
        Args:
            file_path: Ruta al archivo .txt
        
        Returns:
            dict: Características extraídas
        """
        # Leer archivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except:
            raise ValueError(f"No se pudo leer el archivo: {file_path}")
        
        if len(text.strip()) < 10:
            raise ValueError("El archivo contiene muy poco texto para analizar")
        
        # Usar el cargador de transcripciones para análisis
        loader = TranscriptionLoader()
        
        # Detectar idioma
        language = loader.detect_language_content(text)
        
        # Analizar indicadores de demencia
        dementia_indicators = loader.analyze_dementia_indicators(text)
        
        # Calcular métricas del discurso
        discourse_metrics = loader.calculate_discourse_metrics(text)
        
        # Calcular métricas básicas
        words = text.split()
        unique_words = set(word.lower() for word in words if len(word) > 0)

        # Extraer características lingüísticas adicionales
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Calcular repeticiones
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words) if len(words) > 0 else 0

        return {
            'texto': text,
            'texto_limpio': text,  # Preservamos errores
            'idioma': language,
            'num_palabras': len(words),
            'riqueza_lexica': len(unique_words) / len(words) if len(words) > 0 else 0,

            # Características lingüísticas
            'num_sentences': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'num_questions': text.count('?'),
            'num_exclamations': text.count('!'),
            'repetition_ratio': repetition_ratio,

            # Indicadores de demencia
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
            'topic_maintenance': discourse_metrics['topic_maintenance']
        }
    
    def get_user_demographics(self):
        """Solicita información demográfica al usuario"""
        print("\n📋 Por favor, proporciona la siguiente información demográfica:")
        
        try:
            edad = int(input("Edad del paciente: "))
        except:
            edad = 70  # Valor por defecto
            print(f"   Usando edad por defecto: {edad}")
        
        print("Nivel de educación:")
        print("  1. Primaria")
        print("  2. Secundaria") 
        print("  3. Universitaria")
        try:
            edu_choice = int(input("Selecciona (1-3): "))
            nivel_educacion = ['primaria', 'secundaria', 'universitaria'][edu_choice - 1]
        except:
            nivel_educacion = 'secundaria'
            print(f"   Usando nivel por defecto: {nivel_educacion}")
        
        print("Calidad de sueño:")
        print("  1. Mala")
        print("  2. Regular")
        print("  3. Buena")
        try:
            sleep_choice = int(input("Selecciona (1-3): "))
            calidad_sueno = ['mala', 'regular', 'buena'][sleep_choice - 1]
        except:
            calidad_sueno = 'regular'
            print(f"   Usando calidad por defecto: {calidad_sueno}")
        
        print("Hábitos diarios:")
        print("  1. Sedentario")
        print("  2. Ejercicio regular")
        try:
            habit_choice = int(input("Selecciona (1-2): "))
            habitos_diarios = ['sedentario', 'ejercicio regular'][habit_choice - 1]
        except:
            habitos_diarios = 'sedentario'
            print(f"   Usando hábitos por defecto: {habitos_diarios}")
        
        print("Nivel de entorno social:")
        print("  1. Bajo")
        print("  2. Medio")
        print("  3. Alto")
        try:
            social_choice = int(input("Selecciona (1-3): "))
            nivel_entorno_social = ['bajo', 'medio', 'alto'][social_choice - 1]
        except:
            nivel_entorno_social = 'medio'
            print(f"   Usando entorno social por defecto: {nivel_entorno_social}")
        
        return {
            'edad': edad,
            'nivel_educacion': nivel_educacion,
            'calidad_sueno': calidad_sueno,
            'habitos_diarios': habitos_diarios,
            'nivel_entorno_social': nivel_entorno_social
        }
    
    def predict_single_file(self, file_path, demographics=None):
        """
        Hace predicción para un solo archivo
        
        Args:
            file_path: Ruta al archivo de transcripción
            demographics: Información demográfica (opcional)
        
        Returns:
            dict: Resultados de la predicción
        """
        print(f"\n🔍 Analizando archivo: {file_path}")
        
        # Procesar archivo
        features = self.process_text_file(file_path)
        
        # Obtener demografía si no se proporciona
        if demographics is None:
            demographics = self.get_user_demographics()
        
        # Combinar características
        features.update(demographics)
        
        # Crear DataFrame temporal
        df = pd.DataFrame([features])
        
        # Codificar características categóricas
        categorical_mappings = {
            'nivel_educacion': {'primaria': 1, 'secundaria': 2, 'universitaria': 3},
            'calidad_sueno': {'mala': 1, 'regular': 2, 'buena': 3},
            'habitos_diarios': {'sedentario': 1, 'ejercicio regular': 2},
            'nivel_entorno_social': {'bajo': 1, 'medio': 2, 'alto': 3}
        }
        
        for column, mapping in categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping).fillna(1)
        
        # Preparar características adicionales (debe coincidir con el entrenamiento)
        additional_features = [
            'edad', 'nivel_educacion', 'calidad_sueno', 'habitos_diarios',
            'nivel_entorno_social', 'num_palabras', 'riqueza_lexica',
            'num_sentences', 'avg_sentence_length', 'num_questions',
            'num_exclamations', 'repetition_ratio'
        ]
        
        # Asegurar que todas las características estén presentes
        for feature in additional_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Normalizar características (usando valores aproximados del entrenamiento)
        feature_means = {
            'edad': 75, 'nivel_educacion': 2, 'calidad_sueno': 2, 'habitos_diarios': 1.4,
            'nivel_entorno_social': 2, 'num_palabras': 50, 'riqueza_lexica': 0.6,
            'num_sentences': 5, 'avg_sentence_length': 8, 'num_questions': 1,
            'num_exclamations': 0.5, 'repetition_ratio': 0.1
        }

        feature_stds = {
            'edad': 10, 'nivel_educacion': 0.8, 'calidad_sueno': 0.8, 'habitos_diarios': 0.5,
            'nivel_entorno_social': 0.8, 'num_palabras': 30, 'riqueza_lexica': 0.2,
            'num_sentences': 3, 'avg_sentence_length': 5, 'num_questions': 1,
            'num_exclamations': 0.5, 'repetition_ratio': 0.1
        }
        
        for feature in additional_features:
            mean_val = feature_means.get(feature, 0)
            std_val = feature_stds.get(feature, 1)
            df[feature] = (df[feature] - mean_val) / std_val
        
        # Tokenizar texto
        text = features['texto']
        encoding = self.preprocessor.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Preparar inputs
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        additional_feat = torch.tensor(df[additional_features].values, dtype=torch.float).to(self.device)
        
        # Hacer predicción
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, additional_feat)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        # Interpretar resultados
        prob_no_dementia = probabilities[0][0].item()
        prob_dementia = probabilities[0][1].item()
        predicted_class = prediction[0].item()
        
        return {
            'archivo': file_path,
            'idioma_detectado': features['idioma'],
            'num_palabras': features['num_palabras'],
            'prediccion': 'Riesgo de Demencia' if predicted_class == 1 else 'Sin Riesgo Aparente',
            'probabilidad_sin_riesgo': prob_no_dementia,
            'probabilidad_con_riesgo': prob_dementia,
            'confianza': max(prob_no_dementia, prob_dementia),
            'indicadores_clave': {
                'problemas_audio': features['audio_comprehension_issues'],
                'patrones_repetitivos': features['repetitive_patterns'],
                'muletillas': features['filler_words_count'],
                'errores_semanticos': features['semantic_errors'],
                'errores_fonémicos': features['phonemic_errors']
            }
        }
    
    def print_results(self, results):
        """Imprime los resultados de manera legible"""
        print(f"\n" + "="*60)
        print(f"🧠 RESULTADOS DEL ANÁLISIS DE DEMENCIA")
        print(f"="*60)
        
        print(f"📁 Archivo: {results['archivo']}")
        print(f"🌍 Idioma detectado: {results['idioma_detectado']}")
        print(f"📝 Número de palabras: {results['num_palabras']}")
        
        print(f"\n🎯 PREDICCIÓN PRINCIPAL:")
        print(f"   {results['prediccion']}")
        print(f"   Confianza: {results['confianza']:.1%}")
        
        print(f"\n📊 PROBABILIDADES:")
        print(f"   Sin riesgo aparente: {results['probabilidad_sin_riesgo']:.1%}")
        print(f"   Riesgo de demencia: {results['probabilidad_con_riesgo']:.1%}")
        
        print(f"\n🔍 INDICADORES CLAVE DETECTADOS:")
        indicadores = results['indicadores_clave']
        print(f"   Problemas de comprensión auditiva: {indicadores['problemas_audio']}")
        print(f"   Patrones repetitivos: {indicadores['patrones_repetitivos']}")
        print(f"   Muletillas y pausas: {indicadores['muletillas']}")
        print(f"   Errores semánticos: {indicadores['errores_semanticos']}")
        print(f"   Errores fonémicos: {indicadores['errores_fonémicos']}")
        
        # Interpretación
        print(f"\n💡 INTERPRETACIÓN:")
        if results['probabilidad_con_riesgo'] > 0.7:
            print("   ⚠️  ALTO RIESGO: Se detectaron múltiples indicadores de deterioro cognitivo")
        elif results['probabilidad_con_riesgo'] > 0.5:
            print("   ⚡ RIESGO MODERADO: Algunos indicadores presentes, se recomienda evaluación adicional")
        else:
            print("   ✅ BAJO RIESGO: Pocos indicadores de deterioro cognitivo detectados")
        
        print(f"\n⚠️  NOTA: Este es un análisis automatizado preliminar.")
        print(f"   Para diagnóstico definitivo, consulte con un profesional médico.")

def main():
    """Función principal para probar el modelo"""
    print("🧠 PREDICTOR DE DEMENCIA - BERT MULTILINGÜE")
    print("="*50)
    
    # Verificar que existe el modelo
    model_path = "models_test/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo entrenado en: {model_path}")
        print("   Primero ejecuta: python test_training_with_new_data.py")
        return
    
    try:
        # Crear predictor
        predictor = DemenciaPredictor(model_path)
        
        # Solicitar archivo
        print(f"\n📁 Ingresa la ruta del archivo de transcripción (.txt):")
        print(f"   Ejemplo: nueva_transcripcion.txt")
        print(f"   O presiona Enter para usar archivo de ejemplo")
        
        file_path = input("Ruta del archivo: ").strip()
        
        # Usar archivo de ejemplo si no se proporciona
        if not file_path:
            # Crear archivo de ejemplo
            example_text = """Hola, estoy describiendo lo que veo en esta imagen. 
            Veo una cocina donde hay una mujer lavando platos. 
            También hay niños que están... eh... ¿cómo se dice? 
            Tratando de alcanzar unas galletas. 
            El agua se está derramando por el suelo. 
            La mujer no se da cuenta de que el agua... el agua se está saliendo.
            Los niños están parados en una silla que parece que se va a caer.
            Es una situación peligrosa en la cocina."""
            
            file_path = "ejemplo_transcripcion.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(example_text)
            print(f"   Usando archivo de ejemplo creado: {file_path}")
        
        # Verificar que existe el archivo
        if not os.path.exists(file_path):
            print(f"❌ No se encontró el archivo: {file_path}")
            return
        
        # Hacer predicción
        results = predictor.predict_single_file(file_path)
        
        # Mostrar resultados
        predictor.print_results(results)
        
    except Exception as e:
        print(f"❌ Error durante la predicción: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
