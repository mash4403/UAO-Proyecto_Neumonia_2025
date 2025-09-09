#!/bin/bash

# Script para ejecutar tests unitarios del proyecto de neumonia
# Uso: ./run_tests.sh [opciones]

echo "🧪 Ejecutando Tests Unitarios - Proyecto Neumonia 2025"
echo "======================================================"

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [OPCIÓN]"
    echo ""
    echo "OPCIONES:"
    echo "  -a, --all         Ejecutar todos los tests"
    echo "  -v, --verbose     Ejecutar con salida detallada"
    echo "  -c, --coverage    Ejecutar con reporte de cobertura"
    echo "  -f, --fast        Ejecutar tests rápidos solamente"
    echo "  -h, --help        Mostrar esta ayuda"
    echo ""
    echo "EJEMPLOS:"
    echo "  $0 -a             # Todos los tests"
    echo "  $0 -v -c          # Tests detallados con cobertura" 
    echo "  $0 -f             # Solo tests rápidos"
}

# Verificar que UV está disponible
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ Error: UV no está instalado${NC}"
    echo "Instala UV desde: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Función para ejecutar tests
run_tests() {
    local pytest_args="$1"
    local description="$2"
    
    echo -e "${YELLOW}🔄 $description${NC}"
    echo "Comando: PYTHONPATH=. uv run pytest $pytest_args"
    echo ""
    
    if PYTHONPATH=. uv run pytest $pytest_args; then
        echo -e "${GREEN}✅ Tests completados exitosamente${NC}"
        return 0
    else
        echo -e "${RED}❌ Algunos tests fallaron${NC}"
        return 1
    fi
}

# Procesar argumentos
case "${1:-default}" in
    -a|--all)
        run_tests "--tb=short -v" "Ejecutando todos los tests con detalle"
        ;;
    -v|--verbose)
        run_tests "-v -s --tb=long" "Ejecutando tests con salida verbose"
        ;;
    -c|--coverage)
        echo -e "${YELLOW}📊 Ejecutando tests con reporte de cobertura${NC}"
        run_tests "--cov=detector_neumonia --cov-report=term-missing --cov-report=html:htmlcov -v" "Tests con cobertura"
        echo -e "${GREEN}📋 Reporte HTML disponible en: htmlcov/index.html${NC}"
        ;;
    -f|--fast)
        run_tests "-x --tb=short" "Ejecutando tests rápidos (parar en primer fallo)"
        ;;
    -h|--help)
        show_help
        exit 0
        ;;
    default)
        echo -e "${YELLOW}🚀 Ejecutando tests básicos${NC}"
        run_tests "--tb=short" "Tests básicos"
        ;;
    *)
        echo -e "${RED}❌ Opción desconocida: $1${NC}"
        show_help
        exit 1
        ;;
esac

echo ""
echo "======================================================"
echo -e "${GREEN}🎯 Ejecución de tests completada${NC}"
