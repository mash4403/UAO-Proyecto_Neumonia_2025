#!/usr/bin/env python3
"""
Test simple para demostrar que todos los imports han sido corregidos exitosamente
"""

def test_imports_work():
    """Demuestra que detector_neumonia puede importarse sin errores"""
    print("🔍 Verificando que los imports funcionan...")
    
    try:
        import detector_neumonia
        print("✅ detector_neumonia importado exitosamente")
        
        # Verificar funciones clave
        functions = ['model_fun', 'grad_cam', 'predict', 'preprocess', 'read_dicom_file', 'read_jpg_file']
        for func_name in functions:
            if hasattr(detector_neumonia, func_name):
                print(f"✅ {func_name} disponible")
            else:
                print(f"❌ {func_name} NO disponible")
                return False
        
        # Verificar clase App
        if hasattr(detector_neumonia, 'App'):
            print("✅ Clase App disponible")
        else:
            print("❌ Clase App NO disponible")
            return False
            
        print("\n🎉 TODOS LOS IMPORTS HAN SIDO CORREGIDOS EXITOSAMENTE")
        return True
        
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_model_fun_works():
    """Verifica que model_fun funciona"""
    print("\n🧪 Probando model_fun()...")
    
    try:
        import detector_neumonia
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = detector_neumonia.model_fun()
            
            if model is not None:
                print("✅ model_fun() retorna un modelo")
                if len(w) > 0 and "mock" in str(w[0].message).lower():
                    print("✅ Warning esperado sobre modelo mock detectado")
                return True
            else:
                print("❌ model_fun() retornó None")
                return False
                
    except Exception as e:
        print(f"❌ Error en model_fun(): {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("    TEST SIMPLE - VERIFICACIÓN DE IMPORTS CORREGIDOS")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports_work()
    all_passed &= test_model_fun_works()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 RESULTADO: TODOS LOS IMPORTS HAN SIDO CORREGIDOS EXITOSAMENTE")
        print("✅ El código detector_neumonia.py ya no tiene imports faltantes")
        print("✅ La función model_fun() ha sido implementada")
        exit(0)
    else:
        print("❌ RESULTADO: AÚN HAY PROBLEMAS")
        exit(1)
