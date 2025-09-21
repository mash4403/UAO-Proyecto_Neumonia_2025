#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de verificación rápida para confirmar configuración CPU-only.
Útil para CI/CD y deployment checks.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def verify_cpu_setup():
    """Verificación completa del setup CPU."""
    print("🔧 TensorFlow CPU Verification")
    print("=" * 40)
    
    checks = []
    
    # 1. Import check
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported: {tf.__version__}")
        checks.append(True)
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        checks.append(False)
        return False
    
    # 2. GPU disabled check
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) == 0:
            print("✅ GPU devices disabled (CPU-only mode)")
            checks.append(True)
        else:
            print(f"⚠️  GPU devices found: {len(gpu_devices)} - may not be CPU-only")
            checks.append(False)
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        checks.append(False)
    
    # 3. CPU devices check
    try:
        cpu_devices = tf.config.list_physical_devices('CPU')
        print(f"✅ CPU devices available: {len(cpu_devices)}")
        checks.append(True)
    except Exception as e:
        print(f"❌ CPU check failed: {e}")
        checks.append(False)
    
    # 4. Basic operation test
    try:
        test_tensor = tf.constant([1, 2, 3, 4, 5])
        result = tf.reduce_sum(test_tensor)
        expected = 15
        if result.numpy() == expected:
            print(f"✅ Basic operations working: sum = {result.numpy()}")
            checks.append(True)
        else:
            print(f"❌ Basic operations failed: expected {expected}, got {result.numpy()}")
            checks.append(False)
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        checks.append(False)
    
    # 5. Detector module test
    try:
        import detector_neumonia_headless as detector
        model = detector.model_fun()
        print("✅ Detector module loads successfully")
        checks.append(True)
    except Exception as e:
        print(f"❌ Detector module test failed: {e}")
        checks.append(False)
    
    # 6. Quick prediction test
    try:
        import numpy as np
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)  # Smaller for quick test
        label, proba, heatmap = detector.predict(test_image)
        if label in ['bacteriana', 'normal', 'viral'] and 0 <= proba <= 100:
            print(f"✅ Prediction pipeline working: {label} ({proba:.1f}%)")
            checks.append(True)
        else:
            print(f"❌ Prediction pipeline failed: invalid result")
            checks.append(False)
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        checks.append(False)
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} checks passed ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED - CPU setup is ready for production!")
        return True
    else:
        print(f"⚠️  {total - passed} checks failed - review configuration")
        return False

if __name__ == "__main__":
    success = verify_cpu_setup()
    sys.exit(0 if success else 1)
