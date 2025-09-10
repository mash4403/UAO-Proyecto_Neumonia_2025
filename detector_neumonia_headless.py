#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Versión headless del detector de neumonía para uso en Docker/servidores.
Sin dependencias de GUI (tkinter, pyautogui, tkcap).
"""

import warnings
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import backend as K

# Enable eager execution for modern TensorFlow compatibility
# Wrap in try-except to handle mocking scenarios or older TF versions
try:
    if hasattr(tf, 'config') and hasattr(tf.config, 'run_functions_eagerly'):
        tf.config.run_functions_eagerly(True)
except (AttributeError, Exception):
    # If config doesn't exist or method fails, continue without eager execution
    pass

import cv2

# DICOM import
import pydicom as dicom


def model_fun():
    """
    Carga y retorna el modelo de CNN entrenado para clasificación de neumonía.
    
    Returns:
        tf.keras.Model: Modelo entrenado para clasificar radiografías en:
                       - 0: bacteriana
                       - 1: normal  
                       - 2: viral
    
    TODO: Implementar la carga del modelo real desde archivo .h5
    """
    # TODO: Descomentar cuando tengas el archivo del modelo
    # return tf.keras.models.load_model('conv_MLP_84.h5')
    
    # Placeholder temporal - crear un modelo mock para testing
    # ELIMINAR ESTA SECCIÓN cuando implementes la carga real del modelo
    warnings.warn(
        "Usando modelo mock temporal. Implementa la carga real del modelo.",
        UserWarning
    )
    
    # Modelo mock simple para evitar errores usando Functional API
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
    
    # Use Functional API to create model for better compatibility
    inputs = Input(shape=(512, 512, 1))
    x = Conv2D(64, (3, 3), activation='relu', name='conv10_thisone')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compilar modelo mock
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model


def grad_cam(array, model=None):
    """Generate Grad-CAM heatmap for the given array and model.
    
    Args:
        array: Input image array
        model: Pre-built keras model (optional, will create new one if None)
    """
    img = preprocess(array)
    
    # Use provided model or create new one
    if model is None:
        model = model_fun()
    
    # Convert to tf.Tensor for gradient tape
    img_tensor = tf.convert_to_tensor(img)
    
    # Get the last convolutional layer
    last_conv_layer = model.get_layer("conv10_thisone")
    
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])
    
    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute gradients of the class output with respect to feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Compute guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize original image and superimpose heatmap
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]


def predict(array):
    """Predict pneumonia class and generate Grad-CAM heatmap.
    
    Args:
        array: Input image array
        
    Returns:
        tuple: (label, probability, heatmap)
    """
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    
    # Make the prediction
    predictions = model.predict(batch_array_img)
    prediction = np.argmax(predictions)
    proba = np.max(predictions) * 100
    
    # Map prediction to label
    label = ""
    if prediction == 0:
        label = "bacteriana"
    elif prediction == 1:
        label = "normal"
    elif prediction == 2:
        label = "viral"
    
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    # Pass the already built model to avoid recreation
    heatmap = grad_cam(array, model)
    
    return (label, proba, heatmap)


def read_dicom_file(path):
    try:
        img = dicom.dcmread(path)
        img_array = img.pixel_array
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        return img_RGB
    except Exception as e:
        # If DICOM reading fails, try to force read or treat as regular image
        try:
            img = dicom.dcmread(path, force=True)
            img_array = img.pixel_array
            img2 = img_array.astype(float)
            img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
            img2 = np.uint8(img2)
            img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            return img_RGB
        except:
            raise Exception(f"No se pudo leer el archivo DICOM: {str(e)}")


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


def predict_from_file(image_path, output_path=None):
    """
    Función para predecir desde archivo de imagen.
    Útil para APIs o procesamiento batch.
    
    Args:
        image_path (str): Ruta al archivo de imagen
        output_path (str, optional): Ruta para guardar el heatmap
    
    Returns:
        dict: Resultado de la predicción
    """
    # Determinar tipo de archivo
    file_ext = image_path.lower().split('.')[-1]
    
    if file_ext == 'dcm':
        img_array = read_dicom_file(image_path)
    else:
        img_array = read_jpg_file(image_path)
    
    # Hacer predicción
    label, proba, heatmap = predict(img_array)
    
    # Guardar heatmap si se especifica
    if output_path:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.title(f'Heatmap - {label} ({proba:.2f}%)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'label': label,
        'probability': proba,
        'prediction_class': label,
        'confidence': proba / 100.0,
        'has_heatmap': True
    }


if __name__ == "__main__":
    print("🏥 Detector de Neumonía - Versión Headless")
    print("Este módulo está diseñado para uso en servidores sin GUI")
    
    # Test básico
    test_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = predict(test_array)
    print(f"Test completado: {result[0]} con {result[1]:.2f}% de confianza")
