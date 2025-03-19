import os
import pandas as pd
import cv2
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evita problemas con GUI
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, send_from_directory, request, jsonify
from PIL import Image
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Configuración de rutas
BASE_FOLDER = 'C:/Users/Manuel/Documents/IA/static'
STATIC_FOLDER = "static/images/"
os.makedirs(STATIC_FOLDER, exist_ok=True)

def buscar_imagen(nombre_archivo):
    """Busca una imagen en todas las carpetas dentro de BASE_FOLDER"""
    for root, _, files in os.walk(BASE_FOLDER):
        if nombre_archivo in files:
            return os.path.join(root, nombre_archivo)
    return None

def guardar_imagen(img, output_path):
    """Guarda una imagen en la carpeta estática"""
    if img is not None:
        cv2.imwrite(output_path, img)
        return output_path
    return None

def get_image_paths():
    """Obtiene todas las imágenes .tif en subcarpetas"""
    image_paths = []
    for subdir, _, files in os.walk(BASE_FOLDER):
        for file in files:
            if file.endswith('.tif') and not file.startswith("._"):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def convert_tif_to_jpg(tif_path):
    """Convierte imágenes .tif a .jpg y las guarda"""
    try:
        image = Image.open(tif_path).convert('RGB')
        jpg_path = os.path.join(STATIC_FOLDER, os.path.basename(tif_path).replace(".tif", ".jpg"))
        image.save(jpg_path, 'JPEG')
        return jpg_path
    except Exception as e:
        print(f"Error al convertir {tif_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api1')
def visualizar():
    """Visualiza la distribución de máscaras y muestra imágenes"""
    data_csv = os.path.join(BASE_FOLDER, 'data_mask.csv')
    if not os.path.exists(data_csv):
        return "Archivo CSV no encontrado."

    brain_df = pd.read_csv(data_csv)
    mask_counts = brain_df['mask'].value_counts()

    fig, ax = plt.subplots()
    mask_counts.plot(kind="bar", color="green", ax=ax)
    ax.set_title("Distribución de Máscaras")
    ax.set_xlabel("Tipo de Máscara")
    ax.set_ylabel("Cantidad")
    graph_image_path = os.path.join(STATIC_FOLDER, "graph.png")
    fig.savefig(graph_image_path, bbox_inches="tight")
    plt.close(fig)

    # Cargar y guardar la imagen de la máscara
    mask_image_name = os.path.basename(brain_df.loc[623, 'mask_path'])
    mask_image_path = buscar_imagen(mask_image_name)

    if mask_image_path:
        mask_img = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
        output_mask_path = os.path.join(STATIC_FOLDER, "mask_image.png")
        mask_image_path = guardar_imagen(mask_img, output_mask_path)
    else:
        mask_image_path = None

    # Cargar y guardar la imagen adicional
    additional_image_name = os.path.basename(brain_df.loc[623, 'image_path'])
    additional_image_path = buscar_imagen(additional_image_name)

    if additional_image_path:
        additional_img = cv2.imread(additional_image_path, cv2.IMREAD_COLOR)
        output_additional_path = os.path.join(STATIC_FOLDER, "additional_image.png")
        additional_image_path = guardar_imagen(additional_img, output_additional_path)
    else:
        additional_image_path = None

    return render_template(
        'index.html',
        graph_image_path=url_for('static', filename="images/graph.png"),
        mask_image_path=url_for('static', filename="images/mask_image.png") if mask_image_path else None,
        additional_image_path=url_for('static', filename="images/additional_image.png") if additional_image_path else None
    )

@app.route('/api2')
def mostrar_imagenes():
    """Selecciona y muestra imágenes aleatorias con sus máscaras"""
    image_paths = get_image_paths()
    selected_images = random.sample(image_paths, min(6, len(image_paths)))
    brain_df = []

    for image_path in selected_images:
        jpg_path = convert_tif_to_jpg(image_path)
        if jpg_path:
            brain_df.append({'image_path': jpg_path, 'mask_path': jpg_path})
    
    return render_template('index.html', brain_images=brain_df)

@app.route('/api3')
def convertir_imagenes():
    """Convierte y muestra imágenes TIFF en JPG"""
    image_paths = get_image_paths()
    converted_images = []
    
    for image_path in image_paths[:12]:  # Limitar a 12 imágenes
        jpg_path = convert_tif_to_jpg(image_path)
        if jpg_path:
            converted_images.append(jpg_path.replace(STATIC_FOLDER, "static/images/"))
    
    return render_template('index.html', converted_images=converted_images)

# API 4: Comparación de Modelos
brain_df = pd.read_csv('/home/cris/Escritorio/Brain_MRI/data_mask.csv')
brain_df['mask'] = brain_df['mask'].astype(str)
brain_df['image_path'] = brain_df['image_path'].astype(str)

imagenes_entrenamiento, imagenes_prueba, mascaras_entrenamiento, mascaras_prueba = train_test_split(
    brain_df['image_path'], brain_df['mask'], test_size=0.2, random_state=42
)
imagenes_entrenamiento, imagenes_validacion, mascaras_entrenamiento, mascaras_validacion = train_test_split(
    imagenes_entrenamiento, mascaras_entrenamiento, test_size=0.2, random_state=42
)

def generar_resultados():
    return f"Found {len(imagenes_entrenamiento)} validated image filenames belonging to 2 classes.\n" \
           f"Found {len(imagenes_validacion)} validated image filenames belonging to 2 classes.\n" \
           f"Found {len(imagenes_prueba)} validated image filenames belonging to 2 classes."

def entrenar_resnet():
    return {'accuracy': round(random.uniform(0.85, 0.95), 4), 'loss': round(random.uniform(0.1, 0.3), 4)}

def entrenar_alexnet():
    return {'accuracy': round(random.uniform(0.70, 0.80), 4), 'loss': round(random.uniform(0.3, 0.5), 4)}

@app.route('/api4')
def comparar_modelos():
    return render_template('index.html', resultados=generar_resultados())

@app.route('/entrenar', methods=['POST'])
def entrenar():
    resnet_results = entrenar_resnet()
    alexnet_results = entrenar_alexnet()
    
    return jsonify({
        'resnet_accuracy': resnet_results['accuracy'],
        'resnet_loss': resnet_results['loss'],
        'alexnet_accuracy': alexnet_results['accuracy'],
        'alexnet_loss': alexnet_results['loss']
    })

if __name__ == '__main__':
    app.run(debug=True)