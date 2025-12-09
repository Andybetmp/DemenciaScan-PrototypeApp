"""
Script de entrenamiento para BERT multilingüe
Detección de demencia con fine-tuning y validación
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import warnings
from datetime import datetime

# Importar módulos locales
from data_preprocessing import DemenciaPreprocessor
from bert_model import MultimodalBERTClassifier, BERTEnsemble, DemenciaLoss, create_model_config, count_parameters

warnings.filterwarnings('ignore')

class DemenciaTrainer:
    """Entrenador para modelos BERT de detección de demencia"""
    
    def __init__(self, 
                 model_config,
                 learning_rate=2e-5,
                 weight_decay=0.01,
                 warmup_steps=100,
                 max_grad_norm=1.0,
                 device=None):
        """
        Inicializa el entrenador
        
        Args:
            model_config: Configuración del modelo
            learning_rate: Tasa de aprendizaje
            weight_decay: Regularización L2
            warmup_steps: Pasos de calentamiento
            max_grad_norm: Norma máxima del gradiente
            device: Dispositivo (CPU/GPU)
        """
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        
        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"🖥️  Usando dispositivo: {self.device}")
        
        # Inicializar modelo
        self.model = MultimodalBERTClassifier(**model_config)
        self.model.to(self.device)
        
        # Información del modelo
        param_info = count_parameters(self.model)
        print(f"📊 Parámetros del modelo: {param_info['trainable_parameters']:,}")
        
        # Configurar optimizador
        self.optimizer = self._setup_optimizer()
        
        # Configurar función de pérdida
        self.criterion = DemenciaLoss(use_focal=True, focal_alpha=0.25, focal_gamma=2.0)
        
        # Métricas de entrenamiento
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Mejor modelo
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def _setup_optimizer(self):
        """Configura el optimizador con diferentes learning rates"""
        # Diferentes learning rates para BERT y capas personalizadas
        bert_params = []
        custom_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                custom_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': self.learning_rate},
            {'params': custom_params, 'lr': self.learning_rate * 10}  # LR más alto para capas nuevas
        ], weight_decay=self.weight_decay)
        
        return optimizer
    
    def _setup_scheduler(self, num_training_steps):
        """Configura el scheduler de learning rate"""
        from transformers import get_linear_schedule_with_warmup
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_epoch(self, train_loader):
        """Entrena una época"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Entrenando")
        
        for batch in progress_bar:
            # Mover datos al dispositivo
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            additional_features = batch['additional_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, additional_features)
            logits = outputs['logits']
            
            # Calcular pérdida
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Actualizar pesos
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Métricas
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calcular métricas de época
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self, val_loader):
        """Valida una época"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando"):
                # Mover datos al dispositivo
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, additional_features)
                logits = outputs['logits']
                
                # Calcular pérdida
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Predicciones y probabilidades
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calcular métricas
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # AUC-ROC si hay suficientes clases
        try:
            auc_roc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        except:
            auc_roc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self, train_loader, val_loader, num_epochs=10, save_dir='models'):
        """
        Entrena el modelo completo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
            save_dir: Directorio para guardar modelos
        """
        print(f"🚀 Iniciando entrenamiento por {num_epochs} épocas...")
        
        # Crear directorio de modelos
        os.makedirs(save_dir, exist_ok=True)
        
        # Configurar scheduler
        num_training_steps = len(train_loader) * num_epochs
        self.scheduler = self._setup_scheduler(num_training_steps)
        
        # Entrenamiento
        for epoch in range(num_epochs):
            print(f"\n📅 Época {epoch + 1}/{num_epochs}")
            
            # Entrenar
            train_metrics = self.train_epoch(train_loader)
            
            # Validar
            val_metrics = self.validate_epoch(val_loader)
            
            # Guardar métricas
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['train_f1'].append(train_metrics['f1'])
            self.train_history['val_f1'].append(val_metrics['f1'])
            
            # Imprimir métricas
            print(f"📊 Entrenamiento - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"📊 Validación - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc_roc']:.4f}")
            
            # Guardar mejor modelo
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                
                # Guardar modelo
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'model_config': self.model_config,
                    'val_f1': self.best_val_f1,
                    'epoch': epoch + 1,
                    'train_history': self.train_history
                }, model_path)
                
                print(f"💾 Mejor modelo guardado (F1: {self.best_val_f1:.4f})")
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"🏆 Mejor F1 de validación: {self.best_val_f1:.4f}")
        
        # Cargar mejor modelo
        self.model.load_state_dict(self.best_model_state)
        
        return self.train_history
    
    def plot_training_history(self, save_path='training_history.png'):
        """Grafica el historial de entrenamiento"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_history['train_loss'], label='Entrenamiento', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validación', color='red')
        axes[0, 0].set_title('Pérdida por Época')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Pérdida')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_history['train_acc'], label='Entrenamiento', color='blue')
        axes[0, 1].plot(self.train_history['val_acc'], label='Validación', color='red')
        axes[0, 1].set_title('Precisión por Época')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Precisión')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.train_history['train_f1'], label='Entrenamiento', color='blue')
        axes[1, 0].plot(self.train_history['val_f1'], label='Validación', color='red')
        axes[1, 0].set_title('F1-Score por Época')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if hasattr(self, 'scheduler'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs, color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Paso')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Gráficas guardadas en: {save_path}")

def main():
    """Función principal de entrenamiento"""
    print("🚀 Iniciando entrenamiento de BERT multilingüe para detección de demencia...")
    
    # Configuración
    config = {
        'csv_path': 'transcripciones_procesadas.csv',
        'model_type': 'multilingual',  # 'multilingual', 'spanish', 'english'
        'batch_size': 4,  # Reducido para datasets pequeños
        'max_length': 512,
        'num_epochs': 20,  # Más épocas para datasets pequeños
        'learning_rate': 2e-5,
        'save_dir': 'models'
    }
    
    try:
        # 1. Preparar datos
        print("📊 Preparando datos...")
        preprocessor = DemenciaPreprocessor(
            model_name=create_model_config(config['model_type'])['model_name']
        )
        
        data = preprocessor.prepare_data(config['csv_path'])
        dataloaders = preprocessor.create_dataloaders(
            data, 
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )
        
        # 2. Crear modelo y entrenador
        print("🤖 Creando modelo...")
        model_config = create_model_config(config['model_type'])
        trainer = DemenciaTrainer(
            model_config=model_config,
            learning_rate=config['learning_rate']
        )
        
        # 3. Entrenar
        print("🎯 Iniciando entrenamiento...")
        history = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config['num_epochs'],
            save_dir=config['save_dir']
        )
        
        # 4. Visualizar resultados
        trainer.plot_training_history('training_history.png')
        
        # 5. Guardar configuración
        config_path = os.path.join(config['save_dir'], 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Entrenamiento completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
