import os
import time
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
from datetime import datetime
from tkinter import filedialog

cwd = os.getcwd()

# Config parameters
ECCENTRICITY_THRESHOLD = 0.6
DIAMETER_THRESHOLD = (5, 15)
FACTOR_CONVERSION = 0.052
PATH = cwd+"\images"

def ellipsis_detector(sam_mask):
    ellipsis = []
    thresholded = sam_mask['segmentation'].astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipsis.append({
                'eccentricity': eccentricity(ellipse),
                'ellipse': ellipse,
                'diametro_equivalente': diametro_equivalente(ellipse, FACTOR_CONVERSION)
            })
        except Exception as e:
            print(e, 'No se ha encontrado radio de la naranja')
    return ellipsis

def eccentricity(ellipse):
    (xc, yc), (d1, d2), angle = ellipse
    return np.sqrt(1 - np.power(d1, 2) / np.power(d2, 2))

def diametro_equivalente(ellipse, factor_conversion):
    (xc, yc), (d1, d2), angle = ellipse
    return np.sqrt(d1 * d2) * factor_conversion

# Load SAM
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Monitor de la carpeta de imágenes
directory = filedialog.askdirectory()
#folder_path = r'H:\Alberto\SegmentAnything\PruebaNaranjasConteo\prueba'

folder_path = directory
processed_images = set()

try:
    while True:
        # Obtener la lista de archivos en la carpeta
        files = os.listdir(folder_path)
        
        # Filtrar solo archivos de imagen (puedes ajustar según tus extensiones)
        image_files = [f for f in files if f.endswith('.tiff') or f.endswith('.jpg') or f.endswith('.png')]
        
        # Encontrar nuevas imágenes no procesadas
        new_images = [image for image in image_files if image not in processed_images]
        
        if new_images:
            for image_name in new_images:
                # Procesar cada imagen nueva
                current_image_path = os.path.join(folder_path, image_name)
                image_bgr = cv2.imread(current_image_path)
                
                if image_bgr is None:
                    print(f"Error: No se puede leer la imagen en {current_image_path}")
                    continue
                
                image_rgb = image_bgr[..., ::-1].copy()
                print(f"Procesando imagen: {current_image_path}")
                
                # Generar máscaras
                masks = mask_generator.generate(image_rgb)
                print(f"Número de máscaras generadas: {len(masks)}")
                
                # Procesar cada máscara
                df = pd.DataFrame(columns=["Diametros", "R", "G", "B"])
                count = 1
                
                for mask in masks:
                    mask['ellipsis'] = ellipsis_detector(mask)
                    
                    for ellipse in mask['ellipsis']:
                        d_low, d_high = DIAMETER_THRESHOLD
                        condition = (ellipse['diametro_equivalente'] > d_low and
                                    ellipse['diametro_equivalente'] < d_high and
                                    ellipse['eccentricity'] < ECCENTRICITY_THRESHOLD)
                        
                        if condition:
                            working_img = image_rgb.copy()
                            color = (0, 255, 0)
                            cv2.ellipse(working_img, ellipse['ellipse'], color, 1, cv2.LINE_AA)
                            
                            x, y, w, h = mask['bbox']
                            segmented = working_img * mask['segmentation'][:, :, np.newaxis]
                            
                            R, G, B = cv2.split(segmented[y:y+h, x:x+w])
                            R_mean = round(np.mean(R[R < 255]))
                            G_mean = round(np.mean(G[G < 255]))
                            B_mean = round(np.mean(B[B < 255]))
                            
                            df.loc[len(df)] = [ellipse['diametro_equivalente'], R_mean, G_mean, B_mean]
                            
                            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            cv2.imwrite(f"Segmentada{count}_{current_datetime}.jpg", segmented[y:y+h, x:x+w][..., ::-1])
                            os.chdir(PATH)
                            count += 1
        
                # Guardar resultados en CSV
                df.to_csv(f'datos_{image_name}.csv', index=False, sep=';', decimal=',')
                
                # Agregar la imagen procesada al conjunto de imágenes procesadas
                processed_images.add(image_name)
        else: 
            print('Esperando imagen...')    

        
        # Pausa antes de revisar nuevamente (por ejemplo, 5 segundos)
        # Puedes ajustar este valor según sea necesario
        time.sleep(2)
        
except KeyboardInterrupt:
    pass