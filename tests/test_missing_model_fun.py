import sys
import types
import importlib
import pytest

def _inject_minimal_stubs():
    # Stub muy mínimo de tensorflow.compat.v1.* para que el import no truene ahí
    tf = types.SimpleNamespace()
    compat_v1 = types.SimpleNamespace()
    compat_v1.disable_eager_execution = lambda: None
    experimental = types.SimpleNamespace()
    experimental.output_all_intermediates = lambda *a, **k: None
    compat_v1.experimental = experimental
    tf.compat = types.SimpleNamespace(v1=compat_v1)
    sys.modules.setdefault("tensorflow", tf)

    # Stubs de módulos que podrían no estar instalados pero se importan
    for m in ["pyautogui", "img2pdf"]:
        if m not in sys.modules:
            sys.modules[m] = types.ModuleType(m)

def test_missing_model_fun_symbol():
    """
    Con el código corregido, el import del módulo debería completar,
    y entonces podemos verificar que model_fun SÍ está definido.
    """
    # Ya no necesitamos stubs porque tenemos TensorFlow real instalado
    sys.modules.pop("detector_neumonia", None)
    
    # Agregar el directorio actual al path para importar
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    mod = importlib.import_module("detector_neumonia")
    # En el código corregido, model_fun() debe estar definido.
    assert hasattr(mod, "model_fun"), "En el código corregido debería existir model_fun()"
