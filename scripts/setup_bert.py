"""
Script de instalación rápida para BERT Multilingüe - Detección de Demencia
Automatiza la configuración del entorno y verificación de dependencias
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        if e.stdout:
            print(f"   Salida: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado. Se requiere Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def create_virtual_environment():
    """Crea entorno virtual"""
    venv_name = "venv_bert"
    
    if os.path.exists(venv_name):
        print(f"⚠️  El entorno virtual '{venv_name}' ya existe")
        response = input("¿Deseas recrearlo? (s/n): ").lower()
        if response == 's':
            if platform.system() == "Windows":
                run_command(f"rmdir /s /q {venv_name}", f"Eliminando {venv_name}")
            else:
                run_command(f"rm -rf {venv_name}", f"Eliminando {venv_name}")
        else:
            print(f"✅ Usando entorno virtual existente")
            return True
    
    return run_command(f"python -m venv {venv_name}", "Creando entorno virtual")

def get_activation_command():
    """Obtiene el comando de activación según el SO"""
    if platform.system() == "Windows":
        return "venv_bert\\Scripts\\activate"
    else:
        return "source venv_bert/bin/activate"

def install_dependencies():
    """Instala dependencias"""
    if not os.path.exists("requirements.txt"):
        print("❌ No se encontró requirements.txt")
        return False
    
    # Comando de instalación según el SO
    if platform.system() == "Windows":
        pip_command = "venv_bert\\Scripts\\pip install -r requirements.txt"
    else:
        pip_command = "venv_bert/bin/pip install -r requirements.txt"
    
    return run_command(pip_command, "Instalando dependencias")

def verify_installation():
    """Verifica que las dependencias estén instaladas correctamente"""
    print("🔍 Verificando instalación...")
    
    # Comando de Python según el SO
    if platform.system() == "Windows":
        python_command = "venv_bert\\Scripts\\python"
    else:
        python_command = "venv_bert/bin/python"
    
    test_script = """
import sys
try:
    import torch
    import transformers
    import pandas
    import numpy
    import sklearn
    print("✅ Todas las dependencias principales están instaladas")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Transformers: {transformers.__version__}")
    print(f"   Pandas: {pandas.__version__}")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)
"""
    
    # Escribir script temporal
    with open("temp_verify.py", "w") as f:
        f.write(test_script)
    
    # Ejecutar verificación
    success = run_command(f"{python_command} temp_verify.py", "Verificando dependencias")
    
    # Limpiar archivo temporal
    try:
        os.remove("temp_verify.py")
    except:
        pass
    
    return success

def check_data_file():
    """Verifica que exista el archivo de datos"""
    data_file = "transcripciones_procesadas.csv"
    if os.path.exists(data_file):
        print(f"✅ Archivo de datos encontrado: {data_file}")
        return True
    else:
        print(f"⚠️  No se encontró el archivo de datos: {data_file}")
        print("   Asegúrate de tener el archivo CSV con las transcripciones")
        return False

def show_next_steps():
    """Muestra los próximos pasos"""
    activation_cmd = get_activation_command()
    
    print("\n" + "="*60)
    print("🎉 INSTALACIÓN COMPLETADA")
    print("="*60)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print(f"1. Activar entorno virtual:")
    print(f"   {activation_cmd}")
    
    print(f"\n2. Navegar al directorio del proyecto:")
    print(f"   cd DemenciaScan")
    
    print(f"\n3. Ejecutar el pipeline completo:")
    print(f"   python main_pipeline.py")
    
    print(f"\n4. O ejecutar paso a paso:")
    print(f"   python data_preprocessing.py")
    print(f"   python train_bert.py")
    print(f"   python evaluate_model.py")
    
    print(f"\n📚 Para más información, consulta:")
    print(f"   - DemenciaScan/README.md")
    print(f"   - TODO.md")
    
    print("\n🚀 ¡Listo para entrenar BERT!")

def main():
    """Función principal de instalación"""
    print("🤖 SETUP BERT MULTILINGÜE - DETECCIÓN DE DEMENCIA")
    print("="*60)
    
    # 1. Verificar Python
    if not check_python_version():
        return
    
    # 2. Crear entorno virtual
    if not create_virtual_environment():
        print("❌ No se pudo crear el entorno virtual")
        return
    
    # 3. Instalar dependencias
    if not install_dependencies():
        print("❌ No se pudieron instalar las dependencias")
        return
    
    # 4. Verificar instalación
    if not verify_installation():
        print("❌ La verificación de dependencias falló")
        return
    
    # 5. Verificar archivo de datos
    check_data_file()
    
    # 6. Mostrar próximos pasos
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Instalación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
