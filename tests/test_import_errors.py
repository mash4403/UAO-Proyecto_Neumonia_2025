import sys
import importlib
import pytest

def test_module_import_now_works():
    """Test que verifica que el módulo ahora puede importarse sin errores"""
    sys.modules.pop("detector_neumonia", None)  # asegurar import limpio
    
    # Agregar el directorio actual al path para importar
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Ahora el import debería funcionar sin errores
    try:
        mod = importlib.import_module("detector_neumonia")
        assert mod is not None, "El módulo debería importarse correctamente"
        print("✅ detector_neumonia importado exitosamente")
    except Exception as e:
        pytest.fail(f"El import debería funcionar ahora, pero falló con: {e}")
