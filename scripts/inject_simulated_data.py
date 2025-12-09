import json
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.absolute()
EXPORTS_DIR = ROOT_DIR / "exports"
HISTORY_FILE = EXPORTS_DIR / "history.json"

# Asegurar que el directorio existe
EXPORTS_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
random.seed(42)

# Categorías y sus características para la simulación
CATEGORIES = {
    "Healthy": {
        "prob_alz_range": (0.01, 0.25),
        "indicators": {
            "problemas_audio": (0, 1),
            "patrones_repetitivos": (0, 1),
            "muletillas": (0, 5),
            "errores_semanticos": (0, 2),
            "errores_fonémicos": (0, 1)
        }
    },
    "Borderline/Dudoso": {
        "prob_alz_range": (0.35, 0.55),
        "indicators": {
            "problemas_audio": (1, 2),
            "patrones_repetitivos": (1, 3),
            "muletillas": (5, 12),
            "errores_semanticos": (2, 5),
            "errores_fonémicos": (0, 2)
        }
    },
    "Alzheimer Leve": {
        "prob_alz_range": (0.60, 0.79),
        "indicators": {
            "problemas_audio": (2, 3),
            "patrones_repetitivos": (3, 6),
            "muletillas": (8, 15),
            "errores_semanticos": (4, 8),
            "errores_fonémicos": (1, 3)
        }
    },
    "Alzheimer Moderado": {
        "prob_alz_range": (0.80, 0.94),
        "indicators": {
            "problemas_audio": (3, 4),
            "patrones_repetitivos": (5, 10),
            "muletillas": (10, 20),
            "errores_semanticos": (6, 12),
            "errores_fonémicos": (2, 5)
        }
    },
    "Alzheimer Severo": {
        "prob_alz_range": (0.95, 0.999),
        "indicators": {
            "problemas_audio": (4, 5),
            "patrones_repetitivos": (8, 15),
            "muletillas": (15, 30),
            "errores_semanticos": (10, 20),
            "errores_fonémicos": (4, 8)
        }
    }
}

# Distribución deseada
DISTRIBUTION = {
    "Healthy": 35,
    "Borderline/Dudoso": 15,
    "Alzheimer Leve": 20,
    "Alzheimer Moderado": 20,
    "Alzheimer Severo": 10
}

def generate_simulated_case(case_id, category):
    params = CATEGORIES[category]
    
    # Probabilidades
    prob_alz = random.uniform(*params["prob_alz_range"])
    prob_healthy = 1.0 - prob_alz
    
    # Predicción
    if prob_alz > 0.5:
        prediccion = "Alzheimer"
    else:
        prediccion = "No Alzheimer"
        
    # Indicadores
    indicadores = {}
    for key, (min_val, max_val) in params["indicators"].items():
        indicadores[key] = random.randint(min_val, max_val)
        
    # Fecha simulada (últimos 3 meses)
    days_ago = random.randint(0, 90)
    fecha = (datetime.now() - timedelta(days=days_ago)).isoformat()
    
    return {
        "archivo": f"simulated_case_{case_id:03d}.txt",
        "prediccion": prediccion,
        "probabilidad_sin_riesgo": prob_healthy,
        "probabilidad_con_riesgo": prob_alz,
        "confianza": max(prob_healthy, prob_alz),
        "indicadores_clave": indicadores,
        "fecha": fecha,
        "simulated_category": category # Metadata extra, no afecta la app
    }

def main():
    print("🚀 Iniciando inyección de datos simulados...")
    
    # Cargar historial existente
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"📚 Historial existente cargado: {len(history)} registros")
        except Exception as e:
            print(f"⚠️ Error al leer historial existente: {e}. Se creará uno nuevo.")
            history = []
    
    # Generar nuevos casos
    new_cases = []
    case_id = 1
    
    for category, count in DISTRIBUTION.items():
        for _ in range(count):
            case = generate_simulated_case(case_id, category)
            new_cases.append(case)
            case_id += 1
            
    # Mezclar casos para que no estén ordenados por categoría
    random.shuffle(new_cases)
    
    # Ordenar por fecha (opcional, pero se ve mejor)
    new_cases.sort(key=lambda x: x['fecha'], reverse=True)
    
    # Agregar al historial
    history.extend(new_cases)
    
    # Guardar
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"✅ Éxito: Se han inyectado {len(new_cases)} casos simulados.")
        print(f"📚 Total de registros en historial: {len(history)}")
        print(f"📁 Archivo guardado en: {HISTORY_FILE}")
    except Exception as e:
        print(f"❌ Error al guardar el historial: {e}")

if __name__ == "__main__":
    main()
