#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración optimizada de TensorFlow para CPU en entornos de producción.
"""

import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)

def configure_tensorflow_cpu():
    """
    Configura TensorFlow para operación optimizada solo en CPU.
    
    Esta función debe llamarse antes de importar/usar cualquier modelo de TensorFlow.
    """
    try:
        # 1. Forzar CPU-only (deshabilitar GPU completamente)
        tf.config.set_visible_devices([], 'GPU')
        logger.info("🔥 GPU devices disabled - CPU-only mode activated")
        
        # 2. Configurar threading para máximo rendimiento CPU
        # 0 = usar todos los cores disponibles
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        logger.info("🚀 CPU threading optimized for all available cores")
        
        # 3. Configurar variables de entorno para optimización adicional
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging verbosity
        os.environ['KMP_BLOCKTIME'] = '1'         # Reduce thread block time
        os.environ['KMP_SETTINGS'] = '1'          # Show OpenMP settings
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores
        
        # 4. Configurar memory growth (importante para CPU)
        cpu_devices = tf.config.list_physical_devices('CPU')
        if cpu_devices:
            logger.info(f"📊 CPU devices found: {len(cpu_devices)}")
        
        # 5. Configurar ejecución eager (opcional pero recomendado)
        if hasattr(tf.config, 'run_functions_eagerly'):
            tf.config.run_functions_eagerly(True)
            logger.info("⚡ Eager execution enabled")
        
        # 6. Verificar configuración
        verify_cpu_configuration()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error configuring TensorFlow CPU: {e}")
        return False

def verify_cpu_configuration():
    """Verifica que la configuración CPU esté correcta."""
    try:
        # Verificar que no hay GPUs visibles
        gpu_devices = tf.config.list_physical_devices('GPU')
        cpu_devices = tf.config.list_physical_devices('CPU')
        
        logger.info(f"🔍 Configuration verification:")
        logger.info(f"   - CPU devices: {len(cpu_devices)}")
        logger.info(f"   - GPU devices: {len(gpu_devices)} (should be 0)")
        logger.info(f"   - TensorFlow version: {tf.__version__}")
        
        # Verificar threading
        inter_threads = tf.config.threading.get_inter_op_parallelism_threads()
        intra_threads = tf.config.threading.get_intra_op_parallelism_threads()
        
        logger.info(f"   - Inter-op threads: {inter_threads}")
        logger.info(f"   - Intra-op threads: {intra_threads}")
        logger.info(f"   - System CPU count: {os.cpu_count()}")
        
        # Test simple
        test_tensor = tf.constant([1, 2, 3, 4, 5])
        result = tf.reduce_sum(test_tensor)
        logger.info(f"✅ CPU test successful: sum([1,2,3,4,5]) = {result.numpy()}")
        
        return len(gpu_devices) == 0  # Success if no GPU visible
        
    except Exception as e:
        logger.error(f"❌ Configuration verification failed: {e}")
        return False

def get_cpu_info():
    """Obtiene información del CPU para diagnóstico."""
    info = {
        'tensorflow_version': tf.__version__,
        'cpu_count': os.cpu_count(),
        'cpu_devices': len(tf.config.list_physical_devices('CPU')),
        'gpu_devices': len(tf.config.list_physical_devices('GPU')),
        'inter_op_threads': tf.config.threading.get_inter_op_parallelism_threads(),
        'intra_op_threads': tf.config.threading.get_intra_op_parallelism_threads(),
        'eager_execution': tf.executing_eagerly()
    }
    
    # Variables de entorno relevantes
    env_vars = ['TF_CPP_MIN_LOG_LEVEL', 'KMP_BLOCKTIME', 'OMP_NUM_THREADS']
    info['environment'] = {var: os.environ.get(var, 'Not set') for var in env_vars}
    
    return info

def benchmark_cpu_performance(iterations=1000):
    """
    Benchmark simple para medir rendimiento CPU.
    
    Args:
        iterations: Número de iteraciones para el benchmark
        
    Returns:
        dict: Métricas de rendimiento
    """
    import time
    
    logger.info(f"🏃 Starting CPU benchmark ({iterations} iterations)...")
    
    # Operaciones matemáticas típicas de ML
    start_time = time.time()
    
    for i in range(iterations):
        # Matrix multiplication (típico en neural networks)
        a = tf.random.normal((100, 100))
        b = tf.random.normal((100, 100))
        c = tf.matmul(a, b)
        
        # Activation function
        activated = tf.nn.relu(c)
        
        # Reduction operation
        result = tf.reduce_mean(activated)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    metrics = {
        'total_time_seconds': total_time,
        'iterations': iterations,
        'avg_time_per_iteration_ms': (total_time / iterations) * 1000,
        'operations_per_second': iterations / total_time,
        'cpu_info': get_cpu_info()
    }
    
    logger.info(f"✅ Benchmark completed:")
    logger.info(f"   - Total time: {total_time:.3f}s")
    logger.info(f"   - Avg per iteration: {metrics['avg_time_per_iteration_ms']:.2f}ms")
    logger.info(f"   - Operations/sec: {metrics['operations_per_second']:.1f}")
    
    return metrics

if __name__ == "__main__":
    # Test de configuración
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 TensorFlow CPU Configuration Test")
    print("=" * 50)
    
    success = configure_tensorflow_cpu()
    
    if success:
        print("\n📊 System Information:")
        info = get_cpu_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("\n🏃 Running benchmark...")
        benchmark_cpu_performance(100)
    else:
        print("❌ Configuration failed")
