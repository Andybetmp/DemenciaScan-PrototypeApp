"""
Script para probar el entrenamiento del modelo BERT con el nuevo dataset
de 47 transcripciones preservando errores gramaticales
"""

import pandas as pd
import numpy as np
import os
import sys

# Importar módulos locales
from data_preprocessing import DemenciaPreprocessor
from bert_model import MultimodalBERTClassifier, create_model_config
from train_bert import DemenciaTrainer

def test_new_dataset():
    """Prueba el nuevo dataset con el pipeline BERT"""
    print("🚀 Probando entrenamiento con el nuevo dataset de 47 transcripciones...")
    
    # Verificar que existe el archivo
    csv_path = "transcripciones_entrenamiento_procesadas.csv"
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró el archivo: {csv_path}")
        print("   Ejecuta primero: python load_training_data.py")
        return
    
    # Cargar y mostrar información del dataset
    df = pd.read_csv(csv_path)
    print(f"📊 Dataset cargado: {len(df)} muestras")
    print(f"   Idiomas: {df['idioma'].value_counts().to_dict()}")
    print(f"   Diagnósticos: {df['diagnostico_alzheimer'].value_counts().to_dict()}")
    
    # Mostrar algunas características específicas de demencia
    dementia_cols = [
        'audio_comprehension_issues', 'repetitive_patterns', 'incomplete_thoughts',
        'filler_words_count', 'language_switching', 'semantic_errors',
        'phonemic_errors', 'word_finding_pauses'
    ]
    
    print(f"\n🧠 INDICADORES DE DEMENCIA PROMEDIO:")
    for col in dementia_cols:
        if col in df.columns:
            print(f"   {col}: {df[col].mean():.2f}")
    
    # Configuración para entrenamiento de prueba
    config = {
        'model_type': 'multilingual',
        'batch_size': 4,  # Pequeño para prueba rápida
        'max_length': 256,  # Reducido para prueba rápida
        'num_epochs': 3,    # Pocas épocas para prueba
        'learning_rate': 2e-5
    }
    
    try:
        print(f"\n🔄 Preparando datos para BERT...")
        
        # Crear preprocesador
        model_config = create_model_config(config['model_type'])
        preprocessor = DemenciaPreprocessor(model_name=model_config['model_name'])
        
        # Preparar datos
        data = preprocessor.prepare_data(csv_path, test_size=0.2, val_size=0.1)
        
        print(f"✅ Datos preparados:")
        print(f"   Entrenamiento: {len(data['train']['texts'])} muestras")
        print(f"   Validación: {len(data['val']['texts'])} muestras") 
        print(f"   Prueba: {len(data['test']['texts'])} muestras")
        
        # Crear dataloaders
        dataloaders = preprocessor.create_dataloaders(
            data,
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )
        
        print(f"\n🤖 Creando modelo BERT multilingüe...")
        
        # Actualizar configuración del modelo con las nuevas características
        model_config['num_additional_features'] = len(data['feature_names'])
        
        # Crear entrenador
        trainer = DemenciaTrainer(
            model_config=model_config,
            learning_rate=config['learning_rate']
        )
        
        print(f"📊 Modelo creado con {len(data['feature_names'])} características adicionales")
        print(f"   Características: {data['feature_names'][:5]}...")  # Mostrar primeras 5
        
        # Entrenar por pocas épocas como prueba
        print(f"\n🎯 Iniciando entrenamiento de prueba ({config['num_epochs']} épocas)...")
        
        history = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config['num_epochs'],
            save_dir='models_test'
        )
        
        print(f"\n✅ Entrenamiento de prueba completado!")
        print(f"🏆 Mejor F1 de validación: {trainer.best_val_f1:.4f}")
        
        # Mostrar métricas finales
        if history['val_f1']:
            final_metrics = {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'train_acc': history['train_acc'][-1],
                'val_acc': history['val_acc'][-1],
                'train_f1': history['train_f1'][-1],
                'val_f1': history['val_f1'][-1]
            }
            
            print(f"\n📈 MÉTRICAS FINALES:")
            for metric, value in final_metrics.items():
                print(f"   {metric}: {value:.4f}")
        
        # Evaluar en conjunto de prueba
        print(f"\n🔍 Evaluando en conjunto de prueba...")
        test_results = trainer.validate_epoch(dataloaders['test'])
        
        print(f"📊 RESULTADOS EN CONJUNTO DE PRUEBA:")
        print(f"   Precisión: {test_results['accuracy']:.4f}")
        print(f"   F1-Score: {test_results['f1']:.4f}")
        print(f"   AUC-ROC: {test_results['auc_roc']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🧪 PRUEBA DE ENTRENAMIENTO CON NUEVO DATASET")
    print("=" * 50)
    
    success = test_new_dataset()
    
    if success:
        print(f"\n🎉 ¡Prueba exitosa!")
        print(f"📁 El modelo se guardó en: models_test/")
        print(f"🔄 Para entrenamiento completo, usa: python train_bert.py")
        print(f"🧠 El modelo aprendió de errores gramaticales como indicadores de demencia")
    else:
        print(f"\n❌ La prueba falló. Revisa los errores arriba.")

if __name__ == "__main__":
    main()
