"""
Modelo BERT Multilingüe para Detección de Demencia
Combina características textuales y demográficas para clasificación binaria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

class MultimodalBERTClassifier(nn.Module):
    """
    Modelo BERT multilingüe con características adicionales
    para detección de demencia
    """
    
    def __init__(self, 
                 model_name='bert-base-multilingual-cased',
                 num_additional_features=12,
                 num_classes=2,
                 dropout_rate=0.3,
                 hidden_size=256):
        """
        Inicializa el modelo multimodal
        
        Args:
            model_name: Nombre del modelo BERT preentrenado
            num_additional_features: Número de características adicionales
            num_classes: Número de clases (2 para binario)
            dropout_rate: Tasa de dropout
            hidden_size: Tamaño de capas ocultas
        """
        super(MultimodalBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Cargar BERT preentrenado
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_config = AutoConfig.from_pretrained(model_name)
        
        # Congelar algunas capas de BERT (opcional)
        # self._freeze_bert_layers(num_layers_to_freeze=6)
        
        # Dimensiones
        bert_hidden_size = self.bert_config.hidden_size
        
        # Capas para procesar características adicionales
        self.additional_features_layer = nn.Sequential(
            nn.Linear(num_additional_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Capa de atención para características textuales
        self.text_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Capas de fusión multimodal
        fusion_input_size = bert_hidden_size + hidden_size // 4
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Capa de clasificación final
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # Inicialización de pesos
        self._init_weights()
    
    def _freeze_bert_layers(self, num_layers_to_freeze=6):
        """Congela las primeras capas de BERT"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.bert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def _init_weights(self):
        """Inicializa pesos de capas personalizadas"""
        modules_to_init = [self.additional_features_layer, self.fusion_layers]
        
        for module in modules_to_init:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        # Inicializar classifier por separado
        if isinstance(self.classifier, nn.Linear):
            torch.nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                torch.nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, additional_features):
        """
        Forward pass del modelo
        
        Args:
            input_ids: IDs de tokens de BERT
            attention_mask: Máscara de atención
            additional_features: Características adicionales
        
        Returns:
            logits: Logits de clasificación
            attention_weights: Pesos de atención (para interpretabilidad)
        """
        batch_size = input_ids.size(0)
        
        # 1. Procesar texto con BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Obtener representaciones de secuencia
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output        # [batch_size, hidden_size]
        
        # 2. Aplicar atención a las representaciones de secuencia
        attended_output, attention_weights = self.text_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pooling promedio ponderado por atención
        text_representation = torch.mean(attended_output, dim=1)  # [batch_size, hidden_size]
        
        # 3. Procesar características adicionales
        additional_repr = self.additional_features_layer(additional_features)
        
        # 4. Fusión multimodal
        fused_representation = torch.cat([text_representation, additional_repr], dim=1)
        fused_output = self.fusion_layers(fused_representation)
        
        # 5. Clasificación final
        logits = self.classifier(fused_output)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'text_representation': text_representation,
            'additional_representation': additional_repr,
            'fused_representation': fused_output
        }

class BERTEnsemble(nn.Module):
    """
    Ensemble de modelos BERT para mejorar robustez
    """
    
    def __init__(self, model_configs, num_classes=2):
        """
        Inicializa ensemble de modelos
        
        Args:
            model_configs: Lista de configuraciones de modelos
            num_classes: Número de clases
        """
        super(BERTEnsemble, self).__init__()
        
        self.models = nn.ModuleList()
        for config in model_configs:
            model = MultimodalBERTClassifier(**config)
            self.models.append(model)
        
        self.num_models = len(self.models)
        self.num_classes = num_classes
        
        # Pesos para ensemble (aprendibles)
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
    
    def forward(self, input_ids, attention_mask, additional_features):
        """Forward pass del ensemble"""
        outputs = []
        
        for model in self.models:
            output = model(input_ids, attention_mask, additional_features)
            outputs.append(output['logits'])
        
        # Combinar predicciones con pesos aprendibles
        stacked_outputs = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Promedio ponderado
        ensemble_logits = torch.sum(stacked_outputs * weights.view(-1, 1, 1), dim=0)
        
        return {
            'logits': ensemble_logits,
            'individual_outputs': outputs,
            'ensemble_weights': weights
        }

class DemenciaLoss(nn.Module):
    """
    Función de pérdida personalizada para detección de demencia
    Combina CrossEntropy con regularización
    """
    
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0, use_focal=True):
        """
        Inicializa la función de pérdida
        
        Args:
            class_weights: Pesos para balancear clases
            focal_alpha: Parámetro alpha de Focal Loss
            focal_gamma: Parámetro gamma de Focal Loss
            use_focal: Si usar Focal Loss en lugar de CrossEntropy
        """
        super(DemenciaLoss, self).__init__()
        
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def focal_loss(self, inputs, targets):
        """Implementación de Focal Loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, logits, targets):
        """Calcula la pérdida"""
        if self.use_focal:
            return self.focal_loss(logits, targets)
        else:
            return self.ce_loss(logits, targets)

def create_model_config(model_type='multilingual'):
    """
    Crea configuraciones predefinidas para diferentes tipos de modelos
    
    Args:
        model_type: Tipo de modelo ('multilingual', 'spanish', 'english', 'ensemble')
    
    Returns:
        dict: Configuración del modelo
    """
    configs = {
        'multilingual': {
            'model_name': 'bert-base-multilingual-cased',
            'num_additional_features': 12,
            'num_classes': 2,
            'dropout_rate': 0.3,
            'hidden_size': 256
        },
        'spanish': {
            'model_name': 'dccuchile/bert-base-spanish-wwm-cased',
            'num_additional_features': 12,
            'num_classes': 2,
            'dropout_rate': 0.3,
            'hidden_size': 256
        },
        'english': {
            'model_name': 'distilbert-base-uncased',
            'num_additional_features': 12,
            'num_classes': 2,
            'dropout_rate': 0.3,
            'hidden_size': 256
        },
        'ensemble': [
            {
                'model_name': 'bert-base-multilingual-cased',
                'num_additional_features': 12,
                'num_classes': 2,
                'dropout_rate': 0.2,
                'hidden_size': 256
            },
            {
                'model_name': 'dccuchile/bert-base-spanish-wwm-cased',
                'num_additional_features': 12,
                'num_classes': 2,
                'dropout_rate': 0.3,
                'hidden_size': 256
            }
        ]
    }
    
    return configs.get(model_type, configs['multilingual'])

def count_parameters(model):
    """Cuenta parámetros del modelo"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }

def main():
    """Función principal para probar el modelo"""
    print("🚀 Probando arquitectura del modelo BERT multilingüe...")
    
    # Crear modelo
    config = create_model_config('multilingual')
    model = MultimodalBERTClassifier(**config)
    
    # Contar parámetros
    param_info = count_parameters(model)
    print(f"📊 Información del modelo:")
    print(f"   Parámetros totales: {param_info['total_parameters']:,}")
    print(f"   Parámetros entrenables: {param_info['trainable_parameters']:,}")
    print(f"   Parámetros congelados: {param_info['frozen_parameters']:,}")
    
    # Crear datos de prueba
    batch_size = 2
    seq_length = 128
    num_features = 12
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    additional_features = torch.randn(batch_size, num_features)
    
    # Forward pass
    print(f"\n🔄 Realizando forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, additional_features)
    
    print(f"✅ Forward pass exitoso!")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Attention weights shape: {outputs['attention_weights'].shape}")
    
    # Probar ensemble
    print(f"\n🔄 Probando modelo ensemble...")
    ensemble_config = create_model_config('ensemble')
    ensemble_model = BERTEnsemble(ensemble_config)
    
    with torch.no_grad():
        ensemble_outputs = ensemble_model(input_ids, attention_mask, additional_features)
    
    print(f"✅ Ensemble exitoso!")
    print(f"   Ensemble logits shape: {ensemble_outputs['logits'].shape}")
    print(f"   Ensemble weights: {ensemble_outputs['ensemble_weights']}")

if __name__ == "__main__":
    main()
