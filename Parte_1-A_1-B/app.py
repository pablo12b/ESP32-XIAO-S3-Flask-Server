import threading
import time
import io
import json
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn.functional as F

app = Flask(__name__)

# ========== CONFIG ESP32 (ajusta a tu red) ==========
ESP32_URL = "http://192.168.18.176:81/stream"  # si tu ESP32 funciona con cv2.VideoCapture, úsalo; sino cambia a requests
CAPTURE_FALLBACK = False  # si VideoCapture no funciona, podrías adaptar a requests streaming
# ====================================================

# Shared frames dict (último frame procesado)
frames = {
    "original": None,
    "fgmask": None,
    "filtered": None,
    # Parte 1-B frames
    "noisy": None,
    "denoised_median": None,
    "denoised_blur": None,
    "denoised_gauss_torch": None,
    "edges_canny": None,
    "edges_canny_smoothed": None,
    "edges_sobel": None,
    "edges_sobel_smoothed": None
}

lock = threading.Lock()
running = True

# ========== Parámetros MOG2 (Parte 1-A) ==========
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = True

backSub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY,
                                             varThreshold=MOG2_VAR_THRESHOLD,
                                             detectShadows=MOG2_DETECT_SHADOWS)

# ========== Parámetros de ruido dinámicos (Parte 1-B, modificables desde UI) ==========
noise_params = {
    "noise_percent": 100,     # % de pixeles potencialmente afectados (0-100)
    "gaussian_mean": 0.0,
    "gaussian_sigma": 20.0,   # stddev
    "speckle_var": 0.02,      # var (porcentaje)
    "apply_speckle": True,
    "apply_gaussian": True,
    "filter_kernel_size": 3   # para median/blur/gaussian: debe ser odd
}

# Utility: crear kernel gaussiano para PyTorch conv2d (se usará como denoiser)
def gaussian_kernel_torch(kernel_size=5, sigma=1.0, channels=3):
    ax = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # out_channels, in_channels/groups, kH, kW
    return kernel

# PyTorch denoise using conv2d (different to the repo's example).
# This performs a per-channel Gaussian smoothing via conv2d (learnable-like but fixed).
def pytorch_gaussian_denoise(img_bgr, kernel_size=5, sigma=1.0, device='cpu'):
    """
    img_bgr: np.uint8 HxWx3
    returns denoised np.uint8 HxWx3
    """
    # Normalize to [0,1] float32
    img_t = torch.from_numpy(img_bgr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    kernel = gaussian_kernel_torch(kernel_size, sigma, channels=img_t.shape[1]).to(device)
    # conv2d expects (batch, out_channels, H, W) = conv(groups=channels) trick
    # We'll use groups=channels to apply per-channel kernel
    padding = kernel_size // 2
    out = F.conv2d(img_t, weight=kernel, bias=None, stride=1, padding=padding, groups=img_t.shape[1])
    out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return out_np

# Add Gaussian noise to color image (only to a percentage of pixels)
def add_gaussian_noise_color(img_bgr, mean=0.0, sigma=20.0, percent=100):
    noisy = img_bgr.astype(np.float32).copy()
    h, w, c = noisy.shape
    total = h * w
    n = int((percent / 100.0) * total)
    if n <= 0:
        return img_bgr.copy()
    # choose positions
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    noise = np.random.normal(mean, sigma, (n, c))
    noisy[ys, xs] += noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Add Speckle noise (multiplicative)
def add_speckle_noise_color(img_bgr, var=0.02, percent=100):
    noisy = img_bgr.astype(np.float32).copy()
    h, w, c = noisy.shape
    total = h * w
    n = int((percent / 100.0) * total)
    if n <= 0:
        return img_bgr.copy()
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    # speckle ~ N(0, var) multiplicative
    speck = np.random.normal(0.0, np.sqrt(var), (n, c))
    noisy[ys, xs] = noisy[ys, xs] + noisy[ys, xs] * speck
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Thread: captura desde la ESP32 y procesa frames (parte A + parte B)
def capture_thread_fn():
    global running
    cap = cv2.VideoCapture(ESP32_URL)
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir VideoCapture({ESP32_URL}). Intentando modo fallback.")
        if CAPTURE_FALLBACK:
            # Aquí podrías añadir la lógica con requests streaming (omitted for brevity)
            pass
        else:
            print("[ERROR] VideoCapture falló. Asegúrate que la URL está disponible.")
            running = False
            return

    prev_time = time.time()
    frame_counter = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_counter += 1
        # Resize para mayor velocidad si quieres (opcional)
        # frame = cv2.resize(frame, (640, 480))

        # --- Parte 1-A: background subtraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = backSub.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        _, fgmask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fg_only = cv2.bitwise_and(frame, frame, mask=fgmask_bin)

        # --- Parte 1-B: generar ruido (según parámetros globales)
        np.random.seed()  # asegurar aleatoriedad por hilo
        np_frame = frame.copy()
        p = noise_params
        noisy = np_frame.copy()
        if p.get("apply_gaussian", True) and p.get("gaussian_sigma", 0) > 0:
            noisy = add_gaussian_noise_color(noisy, mean=p["gaussian_mean"],
                                             sigma=p["gaussian_sigma"],
                                             percent=p["noise_percent"])
        if p.get("apply_speckle", False) and p.get("speckle_var", 0) > 0:
            noisy = add_speckle_noise_color(noisy, var=p["speckle_var"],
                                            percent=p["noise_percent"])

        # --- Aplicar filtros de reducción (cv2)
        k = max(3, (int(p["filter_kernel_size"]) // 2) * 2 + 1)  # ensure odd >=3
        denoised_med = cv2.medianBlur(noisy, k)
        denoised_blur = cv2.blur(noisy, (k, k))
        denoised_gauss = cv2.GaussianBlur(noisy, (k, k), 0)

        # --- Denoise using PyTorch conv2d (gaussian kernel) - diferente al ejemplo anterior
        try:
            # usar sigma proporcional al kernel
            sigma_t = max(0.5, k/3.0)
            denoised_torch = pytorch_gaussian_denoise(noisy, kernel_size=k, sigma=sigma_t)
        except Exception as e:
            print("[WARN] PyTorch denoise fallo:", e)
            denoised_torch = denoised_gauss.copy()

        # --- Edge detection comparatives (with/without smoothing)
        # Canny
        canny_no_smooth = cv2.Canny(noisy, 50, 150)
        canny_smooth = cv2.Canny(denoised_gauss, 50, 150)
        # Sobel (magnitude)
        sobelx = cv2.Sobel(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(np.clip(sobel_mag/np.max(sobel_mag+1e-8)*255.0, 0, 255))
        # Sobel on smoothed
        sobelx_s = cv2.Sobel(cv2.cvtColor(denoised_gauss, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        sobely_s = cv2.Sobel(cv2.cvtColor(denoised_gauss, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag_s = np.sqrt(sobelx_s**2 + sobely_s**2)
        sobel_mag_s = np.uint8(np.clip(sobel_mag_s/np.max(sobel_mag_s+1e-8)*255.0, 0, 255))

        # --- copyTo demo: copiar la región foreground del noisy a un fondo negro
        bg_black = np.zeros_like(frame)
        # cv2.copyTo equivalent in Python is cv2.copyTo or dst = cv2.bitwise_and(src, src, mask=mask)
        fg_copied = cv2.copyTo(noisy, fgmask_bin)  # returns image with noisy only where mask==255

        # --- Guardar frames en el dict protegido
        with lock:
            frames["original"] = frame.copy()
            frames["fgmask"] = fgmask_bin.copy()
            frames["filtered"] = fg_only.copy()

            frames["noisy"] = noisy.copy()
            frames["denoised_median"] = denoised_med.copy()
            frames["denoised_blur"] = denoised_blur.copy()
            frames["denoised_gauss_torch"] = denoised_torch.copy()

            frames["edges_canny"] = canny_no_smooth.copy()
            frames["edges_canny_smoothed"] = canny_smooth.copy()
            frames["edges_sobel"] = sobel_mag.copy()
            frames["edges_sobel_smoothed"] = sobel_mag_s.copy()

        # limit FPS of capture loop (si hace falta)
        time.sleep(0.01)

    cap.release()

# Generador MJPEG desde frames dict (stream separado por nombre)
def mjpeg_generator(frame_key):
    """
    Toma el último frame de frames[frame_key] y lo sirve como MJPEG.
    Si la imagen es single-channel (grayscale), se convertirá a BGR para mostrar.
    """
    while True:
        with lock:
            f = frames.get(frame_key, None)
            if f is None:
                # generar frame vacío mientras inicializa
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "Esperando frame...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                (flag, encoded) = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
                time.sleep(0.1)
                continue
            img = f.copy()
        # si es single-channel lo convertimos
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Superponer un pequeño timestamp para referencia
        ts = time.strftime("%H:%M:%S")
        cv2.putText(img, ts, (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        (flag, encoded) = cv2.imencode('.jpg', img)
        if not flag:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
        time.sleep(0.03)  # ~30 fps máximo en el servidor (ajusta según CPU)

# Rutas de streaming (Parte 1-A / 1-B)
@app.route('/')
def index():
    return render_template('index.html')  # template incluido abajo

@app.route('/stream/<key>')
def stream_key(key):
    # key debe corresponder a frames dict keys, por ejemplo: original, fgmask, filtered,
    # noisy, denoised_median, denoised_blur, denoised_gauss_torch, edges_canny...
    valid = list(frames.keys())
    if key not in valid:
        return "Stream key inválida. Usa una de: " + ", ".join(valid), 400
    return Response(mjpeg_generator(key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint para actualizar parámetros de ruido desde la UI (AJAX)
@app.route('/update_noise', methods=['POST'])
def update_noise():
    data = request.json or {}
    # actualizar campos permitidos
    for k in ["noise_percent", "gaussian_mean", "gaussian_sigma", "speckle_var", "apply_speckle", "apply_gaussian", "filter_kernel_size"]:
        if k in data:
            # casting
            try:
                if isinstance(noise_params[k], bool):
                    noise_params[k] = bool(data[k])
                elif isinstance(noise_params[k], int):
                    noise_params[k] = int(data[k])
                else:
                    noise_params[k] = float(data[k]) if ('.' in str(data[k]) or isinstance(noise_params[k], float)) else int(data[k])
            except Exception:
                # fallback: keep previous
                pass
    return jsonify({"status": "ok", "params": noise_params})

# Endpoint para leer parámetros actuales
@app.route('/get_noise_params')
def get_noise_params():
    return jsonify(noise_params)

# Start capture thread
t = threading.Thread(target=capture_thread_fn, daemon=True)
t.start()

# Graceful shutdown hook (opcional)
import atexit
def shutdown():
    global running
    running = False
    t.join(timeout=1)
atexit.register(shutdown)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
