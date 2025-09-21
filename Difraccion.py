import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ======================================================================
# CONFIGURACIÓN — modifica aquí tus parámetros principales
# ======================================================================
z_max_m  = 200                 # [m] tope deseado del slider (se limita por z_max_recomendado)
N                   = 2048
dx                  = 13e-6    # [m/píxel]
longitud_onda_m     = 532e-9   # [m]
lado_abertura_m     = 100e-6   # [m]
centro_abertura_m   = (0.0, 0.0)
amplitud_interior   = 1.0 + 0.0j
fraccion_z_de_zmax  = 0.4      # z inicial = fracción * z_max_slider
mostrar_abertura    = True

# NUEVO: controles de visualización
zoom_vista          = 3.0      # >1: acerca la vista (3× recomendado para z pequeños)
usar_escala_log     = True     # True -> mostrar 10*log10(I/Imax)
rango_dB            = 40.0     # dinámica para la vista log: [-rango_dB, 0] dB
# ======================================================================

# --------------------------- Malla espacial ----------------------------
Lx = N * dx
Ly = N * dx
x = (np.arange(N) - N//2) * dx
y = (np.arange(N) - N//2) * dx
X, Y = np.meshgrid(x, y, indexing="xy")

# --------------------------- Malla de frecuencia -----------------------
f1d_cpm = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
FX_cpm, FY_cpm = np.meshgrid(f1d_cpm, f1d_cpm, indexing="xy")
delta_f_cpm = 1.0/(N*dx)
fny_cpm     = 1.0/(2.0*dx)

# --------------------------- Abertura cuadrada -------------------------
def mascara_abertura_cuadrada(X, Y, lado_m, centro_m=(0.0, 0.0), amplitud_compleja=1.0+0.0j):
    x0_m, y0_m = centro_m
    semi = lado_m/2.0
    dentro = (np.abs(X - x0_m) <= semi) & (np.abs(Y - y0_m) <= semi)
    U0 = np.zeros_like(X, dtype=np.complex128)
    U0[dentro] = amplitud_compleja
    return U0

U0 = mascara_abertura_cuadrada(X, Y, lado_abertura_m, centro_abertura_m, amplitud_interior)

# --------------------------- FFT centrada --------------------------------
def fft2_centrada(u_xy):  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_xy)))
def ifft2_centrada(A_f):  return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A_f)))
A0 = fft2_centrada(U0)

# --------------------------- Propagador ASM ------------------------------
def calcular_kz_radpm(FX_cpm, FY_cpm, lam_m):
    return 2.0*np.pi * np.sqrt((1.0/lam_m**2) - (FX_cpm**2 + FY_cpm**2) + 0j)

def transfer_function_ASM(kz_radpm, z_m):
    return np.exp(1j * z_m * kz_radpm)

kz_radpm = calcular_kz_radpm(FX_cpm, FY_cpm, longitud_onda_m)

# ---- Límite recomendado de z (anti-aliasing) y z inicial
z_max_recomendado_m = (N//2) * (dx**2) / longitud_onda_m
z_max_slider_m      = min(float(z_max_m), float(z_max_recomendado_m))
z_m                 = float(fraccion_z_de_zmax) * z_max_slider_m

# --------------------------- Helpers de visualización --------------------
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]
medio_mm = (N//2) * dx * 1e3
ancho_mm_zoom = medio_mm / float(zoom_vista)

def preparar_para_mostrar(I):
    """
    Devuelve la imagen a mostrar:
    - lineal: I
    - log: 10*log10(I/Imax) con recorte a [-rango_dB, 0]
    """
    if usar_escala_log:
        In = I / (I.max() + 1e-16)
        IdB = 10.0*np.log10(In + 1e-16)
        return np.clip(IdB, -rango_dB, 0.0)
    else:
        return I

# --------------------------- Plots ---------------------------------------
if mostrar_abertura:
    plt.figure()
    plt.title("Abertura cuadrada — amplitud (z = 0)")
    plt.imshow(np.abs(U0), extent=extent_mm)
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.colorbar(label="Amplitud"); plt.tight_layout()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.22)

# Intensidad inicial
H0  = transfer_function_ASM(kz_radpm, z_m)
U_z = ifft2_centrada(A0 * H0)
I_z = np.abs(U_z)**2
I_show = preparar_para_mostrar(I_z)

im = ax.imshow(I_show, extent=extent_mm)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"ASM — Intensidad {('log(dB)' if usar_escala_log else '|U|^2')}   z = {z_m*1e3:.2f} mm")
# fijamos límites de color si es log
if usar_escala_log: im.set_clim(-rango_dB, 0.0)
# aplicamos zoom visual
ax.set_xlim(-ancho_mm_zoom, ancho_mm_zoom)
ax.set_ylim(-ancho_mm_zoom, ancho_mm_zoom)
cbar = fig.colorbar(im, ax=ax, label=("dB" if usar_escala_log else "Intensidad (u.a.)"))

# Slider de z (mm)
ax_z = plt.axes([0.12, 0.08, 0.76, 0.04])
z_slider = Slider(ax=ax_z, label="z (mm)", valmin=0.0, valmax=z_max_slider_m*1e3, valinit=z_m*1e3)

def on_z_change(_):
    z_now_m = z_slider.val * 1e-3
    H = transfer_function_ASM(kz_radpm, z_now_m)
    Uz = ifft2_centrada(A0 * H)
    I  = np.abs(Uz)**2
    I_disp = preparar_para_mostrar(I)

    im.set_data(I_disp)
    if usar_escala_log:
        im.set_clim(-rango_dB, 0.0)
    else:
        im.set_clim(vmin=float(I_disp.min()), vmax=float(I_disp.max()))

    ax.set_title(f"ASM — Intensidad {('log(dB)' if usar_escala_log else '|U|^2')}   z = {z_now_m*1e3:.2f} mm")
    # mantener el zoom
    ax.set_xlim(-ancho_mm_zoom, ancho_mm_zoom)
    ax.set_ylim(-ancho_mm_zoom, ancho_mm_zoom)
    fig.canvas.draw_idle()

z_slider.on_changed(on_z_change)
plt.show()

# --------------------------- Resumen CLI -----------------------------------
print("=== Resumen ===")
print(f"N = {N}, dx = {dx:.3e} m  ->  Lx = Ly = {Lx:.3e} m")
print(f"λ = {longitud_onda_m:.3e} m, lado = {lado_abertura_m:.3e} m")
print(f"Δf = {delta_f_cpm:.6e} ciclos/m  |  f_Nyquist = {fny_cpm:.6e} ciclos/m")
print(f"z_max (recomendado) = {z_max_recomendado_m:.3e} m  ({z_max_recomendado_m*1e3:.3f} mm)")
print(f"z_max (slider)      = {z_max_slider_m:.3e} m  ({z_max_slider_m*1e3:.3f} mm)")
print(f"z inicial           = {z_m:.3e} m  ({z_m*1e3:.3f} mm)")
print(f"zoom_vista = {zoom_vista}×  |  escala_log = {usar_escala_log}  (rango = {rango_dB} dB)")
