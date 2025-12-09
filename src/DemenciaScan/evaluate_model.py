"""
Script de evaluación para modelos BERT de detección de demencia
Incluye métricas detalladas, visualizaciones y análisis de interpretabilidad
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, classification_report
)
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from tqdm import tqdm
import warnings

# Importar módulos locales
from data_preprocessing import DemenciaPreprocessor
from bert_model import MultimodalBERTClassifier, create_model_config
from train_bert import DemenciaTrainer

warnings.filterwarnings('ignore')

class DemenciaEvaluator:
    """Evaluador completo para modelos de detección de demencia"""
    
    def __init__(self, model_path, device=None):
        """
        Inicializa el evaluador
        
        Args:
            model_path: Ruta al modelo guardado
            device: Dispositivo (CPU/GPU)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo
        self.model, self.model_config, self.training_info = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🤖 Modelo cargado desde: {model_path}")
        print(f"🖥️  Dispositivo: {self.device}")
    
    def _load_model(self, model_path):
        """Carga el modelo guardado"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model_config = checkpoint['model_config']
        model = MultimodalBERTClassifier(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        training_info = {
            'best_val_f1': checkpoint.get('val_f1', 0.0),
            'epoch': checkpoint.get('epoch', 0),
            'train_history': checkpoint.get('train_history', {})
        }
        
        return model, model_config, training_info
    
    def evaluate_dataset(self, dataloader, dataset_name="Test"):
        """
        Evalúa el modelo en un dataset completo
        
        Args:
            dataloader: DataLoader del dataset
            dataset_name: Nombre del dataset para logging
        
        Returns:
            dict: Métricas y predicciones detalladas
        """
        print(f"📊 Evaluando en dataset: {dataset_name}")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_features = []
        all_attention_weights = []
        
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluando {dataset_name}"):
                # Mover datos al dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, additional_features)
                logits = outputs['logits']
                
                # Calcular pérdida
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Predicciones y probabilidades
                predictions = torch.argmax(logits, dim=1)
                probabilities = F.softmax(logits, dim=1)
                
                # Guardar resultados
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_features.extend(additional_features.cpu().numpy())
                
                # Guardar pesos de atención para interpretabilidad
                if 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights'].cpu().numpy()
                    all_attention_weights.extend(attention_weights)
        
        # Calcular métricas
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        # Métricas promedio
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        except:
            auc_roc = 0.0
        
        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'dataset_name': dataset_name,
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_avg': precision_avg,
            'recall_avg': recall_avg,
            'f1_avg': f1_avg,
            'auc_roc': auc_roc,
            'confusion_matrix': cm,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities),
            'features': np.array(all_features),
            'attention_weights': all_attention_weights
        }
        
        return results
    
    def print_detailed_metrics(self, results):
        """Imprime métricas detalladas"""
        print(f"\n📈 MÉTRICAS DETALLADAS - {results['dataset_name']}")
        print("=" * 50)
        
        print(f"🎯 Precisión General: {results['accuracy']:.4f}")
        print(f"📊 F1-Score Promedio: {results['f1_avg']:.4f}")
        print(f"🔄 AUC-ROC: {results['auc_roc']:.4f}")
        print(f"💔 Pérdida: {results['loss']:.4f}")
        
        print(f"\n📋 MÉTRICAS POR CLASE:")
        class_names = ['No Alzheimer', 'Alzheimer']
        for i, class_name in enumerate(class_names):
            if i < len(results['precision_per_class']):
                print(f"  {class_name}:")
                print(f"    Precisión: {results['precision_per_class'][i]:.4f}")
                print(f"    Recall: {results['recall_per_class'][i]:.4f}")
                print(f"    F1-Score: {results['f1_per_class'][i]:.4f}")
                print(f"    Soporte: {results['support_per_class'][i]}")
        
        print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
        print(results['confusion_matrix'])
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Grafica matriz de confusión"""
        plt.figure(figsize=(8, 6))
        
        class_names = ['No Alzheimer', 'Alzheimer']
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Número de Muestras'}
        )
        
        plt.title(f'Matriz de Confusión - {results["dataset_name"]}')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Real')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, results, save_path=None):
        """Grafica curva ROC"""
        if len(np.unique(results['labels'])) < 2:
            print("⚠️  No se puede graficar ROC: solo una clase presente")
            return
        
        fpr, tpr, thresholds = roc_curve(results['labels'], results['probabilities'][:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {results["auc_roc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {results["dataset_name"]}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, results, save_path=None):
        """Grafica curva Precision-Recall"""
        if len(np.unique(results['labels'])) < 2:
            print("⚠️  No se puede graficar PR: solo una clase presente")
            return
        
        precision, recall, thresholds = precision_recall_curve(
            results['labels'], results['probabilities'][:, 1]
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precisión')
        plt.title(f'Curva Precisión-Recall - {results["dataset_name"]}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_feature_importance(self, results, feature_names):
        """Analiza la importancia de características adicionales"""
        if len(results['features']) == 0:
            print("⚠️  No hay características adicionales para analizar")
            return
        
        features = results['features']
        labels = results['labels']
        
        # Calcular correlaciones
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df['label'] = labels
        
        correlations = feature_df.corr()['label'].drop('label').sort_values(key=abs, ascending=False)
        
        # Graficar importancia
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in correlations.values]
        bars = plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('Correlación con Diagnóstico')
        plt.title('Importancia de Características Adicionales')
        plt.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, correlations.values)):
            plt.text(value + 0.01 if value >= 0 else value - 0.01, i, 
                    f'{value:.3f}', va='center', 
                    ha='left' if value >= 0 else 'right')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n🔍 IMPORTANCIA DE CARACTERÍSTICAS:")
        for feature, corr in correlations.items():
            print(f"  {feature}: {corr:.4f}")
    
    def visualize_embeddings(self, results, save_path=None):
        """Visualiza embeddings usando t-SNE"""
        if len(results['features']) < 10:
            print("⚠️  Muy pocas muestras para visualización de embeddings")
            return
        
        # Aplicar t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(results['features'])-1))
        embeddings_2d = tsne.fit_transform(results['features'])
        
        # Crear DataFrame para plotly
        df_viz = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': ['Alzheimer' if l == 1 else 'No Alzheimer' for l in results['labels']],
            'prediction': ['Alzheimer' if p == 1 else 'No Alzheimer' for p in results['predictions']],
            'correct': results['labels'] == results['predictions']
        })
        
        # Crear gráfico interactivo
        fig = px.scatter(
            df_viz, x='x', y='y', 
            color='label',
            symbol='correct',
            title=f'Visualización t-SNE de Características - {results["dataset_name"]}',
            hover_data=['prediction'],
            color_discrete_map={'Alzheimer': 'red', 'No Alzheimer': 'blue'}
        )
        
        fig.update_layout(
            xaxis_title='t-SNE Dimensión 1',
            yaxis_title='t-SNE Dimensión 2',
            legend_title='Etiqueta Real'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_evaluation_report(self, results, output_dir='evaluation_results'):
        """Genera reporte completo de evaluación"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Métricas detalladas
        self.print_detailed_metrics(results)
        
        # Gráficos
        self.plot_confusion_matrix(results, 
            save_path=os.path.join(output_dir, f'confusion_matrix_{results["dataset_name"].lower()}.png'))
        
        self.plot_roc_curve(results,
            save_path=os.path.join(output_dir, f'roc_curve_{results["dataset_name"].lower()}.png'))
        
        self.plot_precision_recall_curve(results,
            save_path=os.path.join(output_dir, f'pr_curve_{results["dataset_name"].lower()}.png'))
        
        # Guardar métricas en JSON
        metrics_dict = {
            'dataset_name': results['dataset_name'],
            'accuracy': float(results['accuracy']),
            'f1_avg': float(results['f1_avg']),
            'precision_avg': float(results['precision_avg']),
            'recall_avg': float(results['recall_avg']),
            'auc_roc': float(results['auc_roc']),
            'loss': float(results['loss']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'per_class_metrics': {
                'precision': results['precision_per_class'].tolist(),
                'recall': results['recall_per_class'].tolist(),
                'f1': results['f1_per_class'].tolist(),
                'support': results['support_per_class'].tolist()
            }
        }
        
        metrics_path = os.path.join(output_dir, f'metrics_{results["dataset_name"].lower()}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\n📁 Reporte guardado en: {output_dir}")

def main():
    """Función principal de evaluación"""
    print("🔍 Iniciando evaluación de modelo BERT para detección de demencia...")
    
    # Configuración
    config = {
        'model_path': 'models/best_model.pth',
        'csv_path': 'transcripciones_procesadas.csv',
        'batch_size': 4,
        'max_length': 512,
        'output_dir': 'evaluation_results'
    }
    
    try:
        # Verificar que existe el modelo
        if not os.path.exists(config['model_path']):
            print(f"❌ No se encontró el modelo en: {config['model_path']}")
            print("   Primero ejecuta train_bert.py para entrenar el modelo")
            return
        
        # 1. Cargar evaluador
        evaluator = DemenciaEvaluator(config['model_path'])
        
        # 2. Preparar datos
        print("📊 Preparando datos de evaluación...")
        
        # Cargar configuración del modelo para obtener el tokenizer correcto
        model_name = evaluator.model_config['model_name']
        preprocessor = DemenciaPreprocessor(model_name=model_name)
        
        data = preprocessor.prepare_data(config['csv_path'])
        dataloaders = preprocessor.create_dataloaders(
            data,
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )
        
        # 3. Evaluar en conjunto de prueba
        test_results = evaluator.evaluate_dataset(dataloaders['test'], "Test")
        
        # 4. Evaluar en conjunto de validación
        val_results = evaluator.evaluate_dataset(dataloaders['val'], "Validation")
        
        # 5. Generar reportes
        evaluator.generate_evaluation_report(test_results, config['output_dir'])
        evaluator.generate_evaluation_report(val_results, config['output_dir'])
        
        # 6. Análisis de características
        if len(data['feature_names']) > 0:
            evaluator.analyze_feature_importance(test_results, data['feature_names'])
        
        # 7. Visualización de embeddings
        evaluator.visualize_embeddings(test_results)
        
        print("✅ Evaluación completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
