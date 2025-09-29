import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

# --- Parámetros físicos ---
wavelength = 633e-9      # longitud de onda (633 nm)
pixel_size = 3.45e-6     # tamaño de pixel [m] (ajusta según tu cámara)
threshold = 0.5          # umbral para binarización
z_default = 0.06         # distancia por defecto (m) si el archivo no tiene número

# --- Función de propagación (Espectro Angular) ---
def angular_spectrum_propagation(Uin, wvl, d, dz):
    M, N = Uin.shape
    k = 2 * np.pi / wvl
    fx = np.fft.fftfreq(N, d=d)
    fy = np.fft.fftfreq(M, d=d)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * dz * np.sqrt(1 - (wvl*FX)**2 - (wvl*FY)**2))
    Uin_f = np.fft.fft2(Uin)
    Uout = np.fft.ifft2(Uin_f * H)
    return Uout

# --- Padding con ceros ---
def zero_pad(img, pad_factor=2):
    M, N = img.shape
    Mp, Np = pad_factor*M, pad_factor*N
    out = np.zeros((Mp, Np), dtype=np.complex64)
    out[Mp//2 - M//2:Mp//2 + M//2,
        Np//2 - N//2:Np//2 + N//2] = img
    return out

# --- Carpeta con imágenes ---
carpeta = os.path.dirname(os.path.abspath(__file__))
imagenes = glob.glob(os.path.join(carpeta, "*.bmp"))

# Carpeta para guardar resultados
out_dir = os.path.join(carpeta, "resultados")
os.makedirs(out_dir, exist_ok=True)

if not imagenes:
    print("⚠️ No se encontraron imágenes BMP en la carpeta:", carpeta)
else:
    for archivo in imagenes:
        nombre = os.path.basename(archivo)
        print(f"\n📷 Procesando: {nombre}")

        # Extraer z del nombre (ej. aguja65.bmp -> 65 mm -> 0.065 m)
        num = ''.join([c for c in nombre if c.isdigit()])
        if num:
            z = float(num) / 1000.0
            print(f"   ➡️ Usando z = {z:.3f} m (según nombre del archivo)")
        else:
            z = z_default
            print(f"   ⚠️ No se encontró número en el nombre, usando z={z:.3f} m por defecto")

        # Cargar imagen
        img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   ❌ No se pudo abrir {archivo}")
            continue

        # Estimación de campo en el sensor
        U0 = np.sqrt(img.astype(np.float32) / 255.0)

        # Padding
        U0_pad = zero_pad(U0, pad_factor=2)

        # Retropropagación
        U_back = angular_spectrum_propagation(U0_pad, wavelength, pixel_size, -z)

        # Amplitud normalizada
        amp = np.abs(U_back)
        amp /= amp.max()

        # Binarización
        binary = (amp > threshold).astype(float)

        # --- Mostrar resultados ---
        plt.figure(figsize=(10,4))
        plt.suptitle(f"{nombre}  |  z = {z:.3f} m", fontsize=12)

        plt.subplot(1,2,1)
        plt.imshow(amp, cmap='gray')
        plt.title("Amplitud normalizada")
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.imshow(binary, cmap='gray')
        plt.title("Transmitancia binarizada")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        # --- Guardar resultados ---
        plt.imsave(os.path.join(out_dir, f"{nombre}_amplitud.png"), amp, cmap='gray')
        plt.imsave(os.path.join(out_dir, f"{nombre}_binaria.png"), binary, cmap='gray')
        print(f"   ✅ Resultados guardados en {out_dir}")
