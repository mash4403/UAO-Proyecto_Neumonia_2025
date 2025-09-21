#!/bin/bash
# Script de instalación para Mac/Linux - UAO Proyecto Neumonía

echo "🍎 Configurando UAO Detector de Neumonía para Mac/Linux..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado. Instálalo desde https://python.org"
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias específicas de Mac
echo "📚 Instalando dependencias para Mac/Linux..."
pip install -r requirements-mac.txt

# Verificar instalaciones críticas
echo "🔍 Verificando instalaciones críticas..."

# Verificar TensorFlow
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} instalado correctamente')" 2>/dev/null || echo "⚠️  Problema con TensorFlow"

# Verificar OpenCV
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__} instalado correctamente')" 2>/dev/null || echo "⚠️  Problema con OpenCV"

# Verificar GUI (tkinter)
python3 -c "import tkinter; print('✅ Tkinter (GUI) disponible')" 2>/dev/null || echo "⚠️  Problema con Tkinter"

echo ""
echo "🎉 Instalación completada para Mac/Linux"
echo ""
echo "Para ejecutar la aplicación:"
echo "1. Activa el entorno: source venv/bin/activate"
echo "2. Ejecuta: python3 detector_neumonia.py"
echo ""
echo "O usa el script directo: ./run_mac.sh"

# Hacer ejecutable
chmod +x setup_mac.sh
