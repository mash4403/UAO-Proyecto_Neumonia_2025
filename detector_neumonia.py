#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time

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
    import warnings
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
        img2show = Image.fromarray(img_array)
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        return img_RGB, img2show
    except Exception as e:
        # If DICOM reading fails, try to force read or treat as regular image
        try:
            img = dicom.dcmread(path, force=True)
            img_array = img.pixel_array
            img2show = Image.fromarray(img_array)
            img2 = img_array.astype(float)
            img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
            img2 = np.uint8(img2)
            img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            return img_RGB, img2show
        except:
            raise Exception(f"No se pudo leer el archivo DICOM: {str(e)}")


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None
        
        # Initialize label and proba to avoid AttributeError
        self.label = None
        self.proba = None
        
        # Initialize image attributes
        self.img1 = None
        self.img2 = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        # Ensure the filepath is properly quoted for systems that may misinterpret spaces
        if filepath:
            try:
                # Check file extension to determine how to read
                file_ext = filepath.lower().split('.')[-1]
                if file_ext == 'dcm':
                    self.array, img2show = read_dicom_file(filepath)
                else:
                    self.array, img2show = read_jpg_file(filepath)
                
                self.img1 = img2show.resize((250, 250), Image.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                self.text_img1.image_create(END, image=self.img1)
                self.button1["state"] = "enabled"
            except Exception as e:
                showinfo(title="Error", message=f"Error al cargar la imagen: {str(e)}")

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        if self.label is None or self.proba is None:
            showinfo(title="Error", message="Por favor, ejecute la predicción antes de guardar los resultados.")
            return
        
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        img = cap.capture(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            # Reset attributes
            self.label = None
            self.proba = None
            self.img1 = None
            self.img2 = None
            self.array = None
            self.button1["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
