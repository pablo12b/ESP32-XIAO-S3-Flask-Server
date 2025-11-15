# app.py
# Flask app — Parte 2: operaciones morfológicas para imágenes médicas
# Fecha: 2025-11-15
from flask import Flask, render_template, request, send_file, Response
import cv2
import numpy as np
import os
import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

app = Flask(__name__)

# Carpeta donde pondrás las 3 imágenes médicas (grayscale recommended)
MEDICAL_DIR = os.path.join(app.static_folder, "medical")

# Tamaños de kernel (se probarán al menos 3 tamaños; incluye ~37)
KERNEL_SIZES = [15, 25, 37]  # puedes modificar o añadir otros tamaños

def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se encontró imagen en {path}")
    # Si la imagen tiene canales, convertir a gris
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def morphological_ops(img_gray, kernel_size):
    """
    Devuelve un dict con:
      - erosion
      - dilation
      - tophat
      - blackhat
      - combined = original + (tophat - blackhat)
    Todas las imágenes son uint8 (0-255).
    """
    # structuring element (square)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    erosion = cv2.erode(img_gray, kernel, iterations=1)
    dilation = cv2.dilate(img_gray, kernel, iterations=1)
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # combined: original + (tophat - blackhat)
    # Convertir a int16 para evitar overflow al sumar/restar, luego clip
    combined = img_gray.astype(np.int16) + (tophat.astype(np.int16) - blackhat.astype(np.int16))
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    return {
        "erosion": erosion,
        "dilation": dilation,
        "tophat": tophat,
        "blackhat": blackhat,
        "combined": combined
    }

def make_collage(original, results_dict, kernel_size):
    """
    Crea una imagen compuesta (collage) con:
    [ Original | Erosion | Dilation ]
    [ TopHat   | BlackHat | Combined ]
    Añade textos con métricas (PSNR, SSIM, Sharpness).
    Devuelve BGR (para visualizar en navegador).
    """
    # Convertir todos a BGR para concatenar y texto
    def g2b(imgg):
        return cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR)

    orig_b = g2b(original)
    er_b = g2b(results_dict["erosion"])
    di_b = g2b(results_dict["dilation"])
    th_b = g2b(results_dict["tophat"])
    bh_b = g2b(results_dict["blackhat"])
    c_b = g2b(results_dict["combined"])

    # Añadir textos métricos en cada panel (PSNR/SSIM/Sharpness respecto original)
    panels = [orig_b, er_b, di_b, th_b, bh_b, c_b]
    titles = ["Original", "Erosion", "Dilation", "TopHat", "BlackHat", "Original + (TopHat-BlackHat)"]
    annotated = []
    for i, p in enumerate(panels):
        # calcular métricas
        # PSNR y SSIM comparando canal gris correspondiente con original
        gray_panel = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        try:
            p_psnr = psnr(original, gray_panel, data_range=255)
        except Exception:
            p_psnr = float('nan')
        try:
            p_ssim = ssim(original, gray_panel, data_range=255)
        except Exception:
            p_ssim = float('nan')
        # Sharpness: variance of Laplacian (may be higher = más nítida)
        lap = cv2.Laplacian(gray_panel, cv2.CV_64F)
        sharp = float(lap.var())

        # colocar textos
        text1 = f"{titles[i]}"
        text2 = f"PSNR:{p_psnr:.1f} SSIM:{p_ssim:.3f}"
        text3 = f"Sharp:{sharp:.1f}"
        p_copy = p.copy()
        cv2.putText(p_copy, text1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(p_copy, text2, (8, p_copy.shape[0]-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(p_copy, text3, (8, p_copy.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)
        annotated.append(p_copy)

    # Concatenar en 2 filas x 3 columnas
    top_row = np.hstack((annotated[0], annotated[1], annotated[2]))
    bot_row = np.hstack((annotated[3], annotated[4], annotated[5]))
    collage = np.vstack((top_row, bot_row))

    # Añadir una banda superior con el kernel info
    h, w = collage.shape[:2]
    banner = np.full((40, w, 3), 30, dtype=np.uint8)
    info_text = f"Kernel: {kernel_size}x{kernel_size}    (Compare nitidez y contraste visualmente)"
    cv2.putText(banner, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1, cv2.LINE_AA)
    collage_with_banner = np.vstack((banner, collage))
    return collage_with_banner

@app.route("/")
def index():
    # listar imágenes disponibles en MEDICAL_DIR
    images = []
    if os.path.exists(MEDICAL_DIR):
        for f in sorted(os.listdir(MEDICAL_DIR)):
            if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                images.append(f)
    return render_template("index.html", images=images, kernels=KERNEL_SIZES)

@app.route("/process_preview")
def process_preview():
    """
    Parámetros GET:
      - img : nombre de la imagen (ej: image1.png)
    Devuelve una página con collages para cada kernel size.
    """
    img_name = request.args.get("img", "")
    if img_name == "":
        return "Especifica ?img=image1.png"
    img_path = os.path.join(MEDICAL_DIR, img_name)
    if not os.path.exists(img_path):
        return f"No existe {img_name} en {MEDICAL_DIR}", 404

    previews = []  # lista de tuples (kernel, url_to_collage)
    for k in KERNEL_SIZES:
        previews.append((k, f"/collage?img={img_name}&k={k}"))
    return render_template("preview.html", img_name=img_name, previews=previews)

@app.route("/collage")
def collage():
    """
    Devuelve la imagen JPEG del collage para una imagen y kernel_size.
    GET params:
      - img : nombre (en static/medical/)
      - k   : kernel_size (int)
    """
    img_name = request.args.get("img", "")
    k = int(request.args.get("k", KERNEL_SIZES[0]))
    img_path = os.path.join(MEDICAL_DIR, img_name)
    if not os.path.exists(img_path):
        return "Imagen no encontrada", 404

    orig = load_image_gray(img_path)
    results = morphological_ops(orig, k)
    coll = make_collage(orig, results, k)

    # Encode to JPEG in memory and return
    _, buf = cv2.imencode('.jpg', coll, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    # Correr en 0.0.0.0:5001 (cambia si quieres)
    app.run(host='0.0.0.0', port=5001, debug=False)
