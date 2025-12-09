"""
Pipeline Principal para BERT Multilingüe - Detección de Demencia
Ejecuta todo el proceso: preprocesamiento, entrenamiento y evaluación
"""

import os
import sys
import argparse
import json
from datetime import datetime
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy', 'scikit-learn',
        'matplotlib', 'seaborn', 'tqdm', 'langdetect'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Faltan las siguientes dependencias: {', '.join(missing_packages)}")
        print("   Instala con: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def run_preprocessing(config):
    """Ejecuta el preprocesamiento de datos"""
    print("\n" + "="*60)
    print("🔄 PASO 1: PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    try:
        from data_preprocessing import DemenciaPreprocessor
        
        # Verificar que existe el archivo CSV
        if not os.path.exists(config['csv_path']):
            print(f"❌ No se encontró el archivo: {config['csv_path']}")
            return False
        
        # Inicializar preprocesador
        preprocessor = DemenciaPreprocessor(
            model_name=config['model_name']
        )
        
        # Preparar datos
        data = preprocessor.prepare_data(config['csv_path'])
        
        # Crear dataloaders
        dataloaders = preprocessor.create_dataloaders(
            data,
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )
        
        print("✅ Preprocesamiento completado exitosamente")
        return {'data': data, 'dataloaders': dataloaders, 'preprocessor': preprocessor}
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {str(e)}")
        return False

def run_training(config, preprocessing_results):
    """Ejecuta el entrenamiento del modelo"""
    print("\n" + "="*60)
    print("🚀 PASO 2: ENTRENAMIENTO DEL MODELO")
    print("="*60)
    
    try:
        from train_bert import DemenciaTrainer
        from bert_model import create_model_config
        
        # Crear configuración del modelo
        model_config = create_model_config(config['model_type'])
        
        # Inicializar entrenador
        trainer = DemenciaTrainer(
            model_config=model_config,
            learning_rate=config['learning_rate']
        )
        
        # Entrenar
        history = trainer.train(
            train_loader=preprocessing_results['dataloaders']['train'],
            val_loader=preprocessing_results['dataloaders']['val'],
            num_epochs=config['num_epochs'],
            save_dir=config['save_dir']
        )
        
        # Visualizar resultados
        trainer.plot_training_history(
            os.path.join(config['save_dir'], 'training_history.png')
        )
        
        print("✅ Entrenamiento completado exitosamente")
        return {'trainer': trainer, 'history': history}
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation(config, preprocessing_results):
    """Ejecuta la evaluación del modelo"""
    print("\n" + "="*60)
    print("📊 PASO 3: EVALUACIÓN DEL MODELO")
    print("="*60)
    
    try:
        from evaluate_model import DemenciaEvaluator
        
        # Verificar que existe el modelo
        model_path = os.path.join(config['save_dir'], 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"❌ No se encontró el modelo entrenado: {model_path}")
            return False
        
        # Inicializar evaluador
        evaluator = DemenciaEvaluator(model_path)
        
        # Evaluar en conjunto de prueba
        test_results = evaluator.evaluate_dataset(
            preprocessing_results['dataloaders']['test'], 
            "Test"
        )
        
        # Evaluar en conjunto de validación
        val_results = evaluator.evaluate_dataset(
            preprocessing_results['dataloaders']['val'], 
            "Validation"
        )
        
        # Generar reportes
        output_dir = os.path.join(config['save_dir'], 'evaluation_results')
        evaluator.generate_evaluation_report(test_results, output_dir)
        evaluator.generate_evaluation_report(val_results, output_dir)
        
        # Análisis de características
        if len(preprocessing_results['data']['feature_names']) > 0:
            evaluator.analyze_feature_importance(
                test_results, 
                preprocessing_results['data']['feature_names']
            )
        
        print("✅ Evaluación completada exitosamente")
        return {'test_results': test_results, 'val_results': val_results}
        
    except Exception as e:
        print(f"❌ Error en evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_pipeline_config(config, save_dir):
    """Guarda la configuración del pipeline"""
    config_path = os.path.join(save_dir, 'pipeline_config.json')
    
    # Convertir a formato serializable
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    serializable_config['execution_time'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"📁 Configuración guardada en: {config_path}")

def main():
    """Función principal del pipeline"""
    parser = argparse.ArgumentParser(description='Pipeline BERT Multilingüe para Detección de Demencia')
    
    # Argumentos de configuración
    parser.add_argument('--csv_path', default='../transcripciones_procesadas.csv',
                       help='Ruta al archivo CSV con datos')
    parser.add_argument('--model_type', default='multilingual', 
                       choices=['multilingual', 'spanish', 'english'],
                       help='Tipo de modelo BERT')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamaño del batch')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Longitud máxima de secuencia')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Tasa de aprendizaje')
    parser.add_argument('--save_dir', default='models',
                       help='Directorio para guardar modelos')
    parser.add_argument('--skip_training', action='store_true',
                       help='Saltar entrenamiento (solo evaluación)')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Saltar evaluación')
    
    args = parser.parse_args()
    
    # Configuración
    config = {
        'csv_path': args.csv_path,
        'model_type': args.model_type,
        'model_name': {
            'multilingual': 'bert-base-multilingual-cased',
            'spanish': 'dccuchile/bert-base-spanish-wwm-cased',
            'english': 'distilbert-base-uncased'
        }[args.model_type],
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'save_dir': args.save_dir
    }
    
    print("🤖 PIPELINE BERT MULTILINGÜE - DETECCIÓN DE DEMENCIA")
    print("="*60)
    print(f"📊 Configuración:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("="*60)
    
    # Verificar dependencias
    if not check_dependencies():
        return
    
    # Crear directorio de salida
    os.makedirs(config['save_dir'], exist_ok=True)
    
    try:
        # 1. Preprocesamiento
        preprocessing_results = run_preprocessing(config)
        if not preprocessing_results:
            print("❌ Pipeline detenido por error en preprocesamiento")
            return
        
        # 2. Entrenamiento (opcional)
        if not args.skip_training:
            training_results = run_training(config, preprocessing_results)
            if not training_results:
                print("❌ Pipeline detenido por error en entrenamiento")
                return
        else:
            print("⏭️  Saltando entrenamiento...")
        
        # 3. Evaluación (opcional)
        if not args.skip_evaluation:
            evaluation_results = run_evaluation(config, preprocessing_results)
            if not evaluation_results:
                print("❌ Pipeline detenido por error en evaluación")
                return
        else:
            print("⏭️  Saltando evaluación...")
        
        # 4. Guardar configuración
        save_pipeline_config(config, config['save_dir'])
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📁 Resultados guardados en: {config['save_dir']}")
        print("📊 Archivos generados:")
        print("   - best_model.pth (modelo entrenado)")
        print("   - training_history.png (gráficas de entrenamiento)")
        print("   - evaluation_results/ (métricas y visualizaciones)")
        print("   - pipeline_config.json (configuración utilizada)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
