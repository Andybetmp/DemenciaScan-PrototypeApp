"""
Prueba básica para verificar que las dependencias funcionan
"""

print("🔄 Probando importaciones básicas...")

try:
    import pandas as pd
    print("✅ pandas importado correctamente")
except ImportError as e:
    print(f"❌ Error importando pandas: {e}")

try:
    import numpy as np
    print("✅ numpy importado correctamente")
except ImportError as e:
    print(f"❌ Error importando numpy: {e}")

try:
    import torch
    print(f"✅ torch importado correctamente - versión: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Error importando torch: {e}")

try:
    from transformers import AutoTokenizer
    print("✅ transformers importado correctamente")
except ImportError as e:
    print(f"❌ Error importando transformers: {e}")

try:
    from langdetect import detect
    print("✅ langdetect importado correctamente")
except ImportError as e:
    print(f"❌ Error importando langdetect: {e}")

print("\n🔄 Probando carga de datos...")

try:
    # Verificar si existe el archivo CSV
    import os
    csv_path = '../transcripciones_procesadas.csv'
    if os.path.exists(csv_path):
        print(f"✅ Archivo CSV encontrado: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Filas: {len(df)}, Columnas: {len(df.columns)}")
        print(f"   Columnas: {list(df.columns)}")
    else:
        print(f"❌ Archivo CSV no encontrado: {csv_path}")
except Exception as e:
    print(f"❌ Error cargando CSV: {e}")

print("\n🔄 Probando tokenizador BERT...")

try:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    test_text = "Hola, esta es una prueba de tokenización."
    tokens = tokenizer(test_text, return_tensors='pt')
    print("✅ Tokenizador BERT funcionando correctamente")
    print(f"   Tokens shape: {tokens['input_ids'].shape}")
except Exception as e:
    print(f"❌ Error con tokenizador BERT: {e}")

print("\n✅ Prueba básica completada!")
