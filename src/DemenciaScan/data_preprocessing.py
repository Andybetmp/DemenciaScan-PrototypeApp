"""
Preprocesamiento de datos para BERT multilingüe (Español/Inglés)
Detección de demencia basada en transcripciones de audio
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from langdetect import detect, DetectorFactory
import torch
from torch.utils.data import Dataset, DataLoader
import re
import warnings

warnings.filterwarnings('ignore')
DetectorFactory.seed = 0  # Para resultados reproducibles

class DemenciaDataset(Dataset):
    """Dataset personalizado para datos de demencia con BERT multilingüe"""
    
    def __init__(self, texts, labels, additional_features, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.additional_features = additional_features
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        additional_feat = self.additional_features[idx]
        
        # Tokenización
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'additional_features': torch.tensor(additional_feat, dtype=torch.float)
        }

class DemenciaPreprocessor:
    """Preprocesador principal para datos de demencia"""
    
    def __init__(self, model_name='bert-base-multilingual-cased'):
        """
        Inicializa el preprocesador
        
        Args:
            model_name: Nombre del modelo BERT a usar
                       - 'bert-base-multilingual-cased' (multilingüe)
                       - 'dccuchile/bert-base-spanish-wwm-cased' (español)
                       - 'distilbert-base-uncased' (inglés)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
    def detect_language(self, text):
        """Detecta el idioma del texto"""
        try:
            return detect(text)
        except:
            return 'unknown'
    
    def clean_text(self, text):
        """Limpia y normaliza el texto"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remover caracteres especiales pero mantener puntuación básica
        text = re.sub(r'[^\w\s\.\,\?\!\;\:]', ' ', text)
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        # Remover espacios al inicio y final
        text = text.strip()
        
        return text
    
    def encode_categorical_features(self, df):
        """Codifica características categóricas"""
        categorical_mappings = {
            'nivel_educacion': {
                'primaria': 1, 'primeria': 1,  # Corregir typo
                'secundaria': 2,
                'universitaria': 3,
                'posgrado': 4
            },
            'calidad_sueno': {
                'mala': 1,
                'regular': 2,
                'buena': 3
            },
            'habitos_diarios': {
                'sedentario': 1,
                'ejercicio regular': 2,
                'muy activo': 3
            },
            'nivel_entorno_social': {
                'bajo': 1,
                'medio': 2,
                'alto': 3
            }
        }
        
        df_encoded = df.copy()
        
        for column, mapping in categorical_mappings.items():
            if column in df_encoded.columns:
                df_encoded[column] = df_encoded[column].map(mapping).fillna(1)
        
        return df_encoded
    
    def extract_linguistic_features(self, text):
        """Extrae características lingüísticas adicionales"""
        if pd.isna(text) or text == "":
            return {
                'num_sentences': 0,
                'avg_sentence_length': 0,
                'num_questions': 0,
                'num_exclamations': 0,
                'repetition_ratio': 0
            }
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        # Calcular repeticiones
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words) if len(words) > 0 else 0
        
        return {
            'num_sentences': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'num_questions': text.count('?'),
            'num_exclamations': text.count('!'),
            'repetition_ratio': repetition_ratio
        }
    
    def prepare_data(self, csv_path, test_size=0.2, val_size=0.1):
        """
        Prepara los datos para entrenamiento
        
        Args:
            csv_path: Ruta al archivo CSV
            test_size: Proporción para conjunto de prueba
            val_size: Proporción para conjunto de validación
        
        Returns:
            dict: Diccionario con datasets de entrenamiento, validación y prueba
        """
        print("🔄 Cargando datos...")
        df = pd.read_csv(csv_path)
        
        print("🔄 Limpiando textos...")
        df['texto_limpio'] = df['texto'].apply(self.clean_text)
        
        print("🔄 Detectando idiomas...")
        df['idioma'] = df['texto_limpio'].apply(self.detect_language)
        
        print("🔄 Extrayendo características lingüísticas...")
        linguistic_features = df['texto_limpio'].apply(self.extract_linguistic_features)
        linguistic_df = pd.DataFrame(linguistic_features.tolist())
        df = pd.concat([df, linguistic_df], axis=1)
        
        print("🔄 Codificando características categóricas...")
        df = self.encode_categorical_features(df)
        
        # Preparar etiquetas
        if df['diagnostico_alzheimer'].dtype == 'object':
            df['diagnostico_alzheimer'] = df['diagnostico_alzheimer'].map({'si': 1, 'no': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            df['diagnostico_alzheimer'] = df['diagnostico_alzheimer'].fillna(0).astype(int)
        
        # Seleccionar características adicionales
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
        
        # Normalizar características numéricas
        df[additional_features] = self.feature_scaler.fit_transform(df[additional_features])
        
        # Dividir datos
        X_text = df['texto_limpio'].values
        X_features = df[additional_features].values
        y = df['diagnostico_alzheimer'].values
        
        # Manejar datasets pequeños
        if len(X_text) < 3:
            print(f"⚠️  Dataset muy pequeño ({len(X_text)} muestras). Usando todas las muestras para entrenamiento.")
            X_text_train, X_text_val, X_text_test = X_text, X_text, X_text
            X_feat_train, X_feat_val, X_feat_test = X_features, X_features, X_features
            y_train, y_val, y_test = y, y, y
        else:
            # División estratificada para datasets más grandes
            try:
                X_text_temp, X_text_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
                    X_text, X_features, y, test_size=test_size, stratify=y, random_state=42
                )
                
                val_size_adjusted = val_size / (1 - test_size)
                X_text_train, X_text_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
                    X_text_temp, X_feat_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
                )
            except ValueError:
                # Si falla la división estratificada, usar división simple
                print("⚠️  No se puede hacer división estratificada. Usando división simple.")
                X_text_temp, X_text_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
                    X_text, X_features, y, test_size=test_size, random_state=42
                )
                
                val_size_adjusted = val_size / (1 - test_size)
                X_text_train, X_text_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
                    X_text_temp, X_feat_temp, y_temp, test_size=val_size_adjusted, random_state=42
                )
        
        print(f"📊 Distribución de datos:")
        print(f"   Entrenamiento: {len(X_text_train)} muestras")
        print(f"   Validación: {len(X_text_val)} muestras")
        print(f"   Prueba: {len(X_text_test)} muestras")
        
        print(f"📊 Distribución de idiomas:")
        print(df['idioma'].value_counts())
        
        print(f"📊 Distribución de etiquetas:")
        print(f"   Alzheimer: {sum(y)} casos")
        print(f"   No Alzheimer: {len(y) - sum(y)} casos")
        
        return {
            'train': {
                'texts': X_text_train,
                'features': X_feat_train,
                'labels': y_train
            },
            'val': {
                'texts': X_text_val,
                'features': X_feat_val,
                'labels': y_val
            },
            'test': {
                'texts': X_text_test,
                'features': X_feat_test,
                'labels': y_test
            },
            'feature_names': additional_features,
            'dataframe': df
        }
    
    def create_dataloaders(self, data_dict, batch_size=8, max_length=512):
        """Crea DataLoaders para PyTorch"""
        
        datasets = {}
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            datasets[split] = DemenciaDataset(
                texts=data_dict[split]['texts'],
                labels=data_dict[split]['labels'],
                additional_features=data_dict[split]['features'],
                tokenizer=self.tokenizer,
                max_length=max_length
            )
            
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0  # Cambiar a 2-4 si tienes múltiples cores
            )
        
        return dataloaders

def main():
    """Función principal para probar el preprocesamiento"""
    print("🚀 Iniciando preprocesamiento de datos para BERT multilingüe...")
    
    # Inicializar preprocesador
    preprocessor = DemenciaPreprocessor(model_name='bert-base-multilingual-cased')
    
    # Preparar datos
    data = preprocessor.prepare_data('transcripciones_procesadas.csv')
    
    # Crear dataloaders
    dataloaders = preprocessor.create_dataloaders(data, batch_size=2)
    
    print("\n✅ Preprocesamiento completado!")
    print("📁 Archivos generados listos para entrenamiento de BERT")
    
    # Mostrar ejemplo de batch
    train_loader = dataloaders['train']
    for batch in train_loader:
        print(f"\n📋 Ejemplo de batch:")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Attention mask shape: {batch['attention_mask'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Additional features shape: {batch['additional_features'].shape}")
        break

if __name__ == "__main__":
    main()
