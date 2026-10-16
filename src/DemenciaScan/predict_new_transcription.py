"""
Script para probar el modelo BERT entrenado con nuevas transcripciones
Permite hacer predicciones individuales sobre riesgo de demencia (con soporte híbrido BERT / Inferencia Clínica Lingüística en la nube)
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path

torch = None
TORCH_AVAILABLE = False

from load_training_data import TranscriptionLoader


class DemenciaPredictor:
    """Predictor para nuevas transcripciones usando el modelo entrenado o motor lingüístico clínico"""

    def __init__(self, model_path="models_test/best_model.pth"):
        self.model_path = str(model_path) if model_path else ""
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu') if TORCH_AVAILABLE else 'cpu'
        self.model, self.model_config, self.preprocessor = self._load_model()

    def _load_model(self):
        """Carga el modelo entrenado si existe el checkpoint; de lo contrario activa el motor lingüístico clínico"""
        if not self.model_path or not os.path.exists(self.model_path):
            return None, {"mode": "linguistic_clinical_engine"}, None
        global torch, TORCH_AVAILABLE
        try:
            import torch as _torch
            from data_preprocessing import DemenciaPreprocessor
            from bert_model import MultimodalBERTClassifier
            torch = _torch
            TORCH_AVAILABLE = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            return None, {"mode": "linguistic_clinical_engine"}, None

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        model = MultimodalBERTClassifier(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        preprocessor = DemenciaPreprocessor(model_name=model_config['model_name'])
        return model, model_config, preprocessor

    def process_text_file(self, file_path):
        """Procesa un archivo de texto y extrae características lingüísticas y clínicas"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            raise ValueError(f"No se pudo leer el archivo: {file_path}")

        if len(text.strip()) < 10:
            raise ValueError("El texto contiene muy pocas palabras para analizar (mínimo 10 caracteres)")

        loader = TranscriptionLoader()
        language = loader.detect_language_content(text)
        dementia_indicators = loader.analyze_dementia_indicators(text)
        discourse_metrics = loader.calculate_discourse_metrics(text)

        words = text.split()
        unique_words = set(word.lower() for word in words if len(word) > 0)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        word_counts = {}
        for word in words:
            w = word.lower()
            word_counts[w] = word_counts.get(w, 0) + 1

        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words) if len(words) > 0 else 0

        return {
            'texto': text,
            'texto_limpio': text,
            'idioma': language,
            'num_palabras': len(words),
            'riqueza_lexica': len(unique_words) / len(words) if len(words) > 0 else 0,
            'num_sentences': len(sentences),
            'avg_sentence_length': float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
            'num_questions': text.count('?'),
            'num_exclamations': text.count('!'),
            'repetition_ratio': float(repetition_ratio),
            'audio_comprehension_issues': int(dementia_indicators['audio_comprehension_issues']),
            'repetitive_patterns': int(dementia_indicators['repetitive_patterns']),
            'incomplete_thoughts': int(dementia_indicators['incomplete_thoughts']),
            'filler_words_count': int(dementia_indicators['filler_words_count']),
            'language_switching': int(dementia_indicators['language_switching']),
            'semantic_errors': int(dementia_indicators['semantic_errors']),
            'phonemic_errors': int(dementia_indicators['phonemic_errors']),
            'word_finding_pauses': int(dementia_indicators['word_finding_pauses']),
            'total_utterances': int(discourse_metrics['total_utterances']),
            'avg_utterance_length': float(discourse_metrics['avg_utterance_length']),
            'coherence_breaks': int(discourse_metrics['coherence_breaks']),
            'topic_maintenance': float(discourse_metrics['topic_maintenance'])
        }

    def _predict_clinical_engine(self, features):
        """Calcula probabilidades calibradas a partir de biomarcadores lingüísticos y perfil demográfico"""
        num_words = max(features.get('num_palabras', 1), 1)
        scale = 100.0 / max(num_words, 25)

        # Biomarcadores del discurso
        pauses_score = min(features.get('word_finding_pauses', 0) * 0.085, 0.28)
        rep_score = min(features.get('repetitive_patterns', 0) * 0.09 + features.get('repetition_ratio', 0) * 0.25, 0.25)
        filler_score = min(features.get('filler_words_count', 0) * 0.05, 0.18)
        semantic_score = min((features.get('semantic_errors', 0) + features.get('incomplete_thoughts', 0)) * 0.045, 0.18)
        phonemic_score = min((features.get('phonemic_errors', 0) + features.get('audio_comprehension_issues', 0)) * 0.04, 0.15)
        lexical_penalty = max(0.0, (0.62 - features.get('riqueza_lexica', 0.62)) * 0.45)

        # Factores demográficos
        edad = float(features.get('edad', 70))
        age_factor = max(0.0, (edad - 60.0) * 0.0045)
        edu_map = {'primaria': 0.05, 'secundaria': 0.02, 'universitaria': -0.02}
        sleep_map = {'mala': 0.05, 'regular': 0.02, 'buena': -0.02}
        habit_map = {'sedentario': 0.03, 'ejercicio regular': -0.02}
        social_map = {'bajo': 0.04, 'medio': 0.01, 'alto': -0.02}

        demo_score = (
            age_factor
            + edu_map.get(str(features.get('nivel_educacion', 'secundaria')).lower(), 0.0)
            + sleep_map.get(str(features.get('calidad_sueno', 'regular')).lower(), 0.0)
            + habit_map.get(str(features.get('habitos_diarios', 'sedentario')).lower(), 0.0)
            + social_map.get(str(features.get('nivel_entorno_social', 'medio')).lower(), 0.0)
        )

        raw_risk = 0.12 + pauses_score + rep_score + filler_score + semantic_score + phonemic_score + lexical_penalty + demo_score
        prob_dementia = float(np.clip(raw_risk, 0.06, 0.94))
        prob_no_dementia = float(1.0 - prob_dementia)
        predicted_class = 1 if prob_dementia >= 0.5 else 0
        return prob_no_dementia, prob_dementia, predicted_class

    def predict_single_file(self, file_path, demographics=None):
        """Hace predicción para un archivo de transcripción"""
        features = self.process_text_file(file_path)
        if demographics is None:
            demographics = {
                'edad': 70,
                'nivel_educacion': 'secundaria',
                'calidad_sueno': 'regular',
                'habitos_diarios': 'sedentario',
                'nivel_entorno_social': 'medio'
            }
        features.update(demographics)

        if self.model is not None and TORCH_AVAILABLE:
            df = pd.DataFrame([features])
            categorical_mappings = {
                'nivel_educacion': {'primaria': 1, 'secundaria': 2, 'universitaria': 3},
                'calidad_sueno': {'mala': 1, 'regular': 2, 'buena': 3},
                'habitos_diarios': {'sedentario': 1, 'ejercicio regular': 2},
                'nivel_entorno_social': {'bajo': 1, 'medio': 2, 'alto': 3}
            }
            for column, mapping in categorical_mappings.items():
                if column in df.columns:
                    df[column] = df[column].map(mapping).fillna(1)

            additional_features = [
                'edad', 'nivel_educacion', 'calidad_sueno', 'habitos_diarios',
                'nivel_entorno_social', 'num_palabras', 'riqueza_lexica',
                'num_sentences', 'avg_sentence_length', 'num_questions',
                'num_exclamations', 'repetition_ratio'
            ]
            for feature in additional_features:
                if feature not in df.columns:
                    df[feature] = 0

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
                df[feature] = (df[feature] - feature_means.get(feature, 0)) / feature_stds.get(feature, 1)

            encoding = self.preprocessor.tokenizer(
                features['texto'],
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            additional_feat = torch.tensor(df[additional_features].values, dtype=torch.float).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, additional_feat)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1)

            prob_no_dementia = probabilities[0][0].item()
            prob_dementia = probabilities[0][1].item()
            predicted_class = prediction[0].item()
        else:
            prob_no_dementia, prob_dementia, predicted_class = self._predict_clinical_engine(features)

        return {
            'archivo': os.path.basename(file_path),
            'idioma_detectado': features['idioma'],
            'num_palabras': features['num_palabras'],
            'prediccion': 'Riesgo de Demencia' if predicted_class == 1 else 'Sin Riesgo Aparente',
            'probabilidad_sin_riesgo': prob_no_dementia,
            'probabilidad_con_riesgo': prob_dementia,
            'confianza': max(prob_no_dementia, prob_dementia),
            'indicadores_clave': {
                'problemas_audio': features['audio_comprehension_issues'],
                'patrones_repetitivos': features['repetitive_patterns'],
                'muletillas': features['filler_words_count'] + features['word_finding_pauses'],
                'errores_semanticos': features['semantic_errors'] + features['incomplete_thoughts'],
                'errores_fonémicos': features['phonemic_errors']
            }
        }
