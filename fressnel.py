import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ======================================================================
# CONFIGURACIÓN — parámetros principales
# ======================================================================
N                   = 1024         # [muestras] por eje (malla cuadrada N×N)
dx0                 = 6.5e-6       # [m/píxel] paso espacial en el plano z=0
lam                 = 532e-9       # [m] longitud de onda
lado_abertura_m     = 200e-6       # [m] lado de la abertura cuadrada
centro_abertura_m   = (0.0, 0.0)   # [m] centro (x0,y0) de la abertura
amplitud_interior   = 1.0 + 0j     # valor complejo dentro de la abertura
z_max_usuario_m     = 0.20         # [m] tope superior del slider (puedes subirlo)
fraccion_z_inicial  = 0.9          # z inicial = fracción * z_min (>= 1.0 si quieres > z_min)
usar_log            = True         # vista en dB para levantar lóbulos débiles
rango_dB            = 35.0         # dinámica en dB (solo si usar_log=True)
zoom_vista          = 2.0          # >1 acerca la vista (solo visual)
# ======================================================================

# ------------------------------- Malla z=0 ------------------------------
x0 = (np.arange(N) - N//2) * dx0
y0 = (np.arange(N) - N//2) * dx0
X0, Y0 = np.meshgrid(x0, y0, indexing="xy")

# --------------------------- Abertura cuadrada --------------------------
def mascara_abertura_cuadrada(X, Y, lado_m, centro_m=(0.0, 0.0), A=1.0+0j):
    xC, yC = centro_m
    h = lado_m/2
    mask = (np.abs(X - xC) <= h) & (np.abs(Y - yC) <= h)
    U = np.zeros_like(X, dtype=np.complex128)
    U[mask] = A
    return U

U0 = mascara_abertura_cuadrada(X0, Y0, lado_abertura_m, centro_abertura_m, amplitud_interior)

# ---------------------- FFT centrada (helpers) --------------------------
def fft2c(u):  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
def ifft2c(U): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

# ------------------ Restricción de Fresnel y z inicial ------------------
# z_min (Fresnel 1-FFT):  z >= (N/2) * dx0^2 / lam
z_min_m = (N//2) * (dx0**2) / lam
z_max_slider_m = max(z_min_m, float(z_max_usuario_m))
z0 = max(z_min_m, fraccion_z_inicial * z_min_m)

# ------------------ Núcleo Fresnel 1-FFT (pasos de la lámina) ----------
# k = 2π/λ y factor cuadrático común
k = 2*np.pi/lam

def fresnel_1fft(U0_xy: np.ndarray, dx0_m: float, lam_m: float, z_m: float):
    """
    Implementa la Transformada de Fresnel discreta (1 FFT) con:
      - Fase parabolica de la primera transformada U' = *U0  exp(i*k/(2z) * (x0^2 + y0^2))
      - Aplicar transformada de fourier con la función (suma Δ^2): U'' = FFT{U'} * (dx0^2)
      - Pultiplicar por las fases y escalar U = (e^{ikz}/(i λ z)) * exp(i*k/(2z)*(x^2+y^2)) * U''
    Devuelve:
      U_xy_z : campo complejo en el plano z, muestreado con Δx_z = λ z / (N Δx0)
      dx_z   : paso espacial del plano z (metros/píxel)
    """
    N = U0_xy.shape[0]
    # malla de entrada
    xv = (np.arange(N) - N//2) * dx0_m
    X, Y = np.meshgrid(xv, xv, indexing="xy")

    # 1) Pre-chirp
    quad_in = np.exp(1j * k/(2*z_m) * (X**2 + Y**2))
    Uprime = U0_xy * quad_in

    # 2) FFT (con factor Δ^2 para aproximar integral)
    U2 = fft2c(Uprime) * (dx0_m**2)

    # 3) Muestreo de salida (depende de z): Δ = λ z / (N Δx0)
    dx_z = lam_m * z_m / (N * dx0_m)
    xz = (np.arange(N) - N//2) * dx_z
    Xz, Yz = np.meshgrid(xz, xz, indexing="xy")

    # 4) Escala + post-chirp
    prefactor = np.exp(1j * k * z_m) / (1j * lam_m * z_m)
    quad_out  = np.exp(1j * k/(2*z_m) * (Xz**2 + Yz**2))

    Uz = prefactor * quad_out * U2
    return Uz, dx_z

# -------------------- Figura + slider (z en mm) -------------------------
def preparar_imagen(I):
    if usar_log:
        In = I / (I.max() + 1e-16)
        IdB = 10*np.log10(In + 1e-16)
        return np.clip(IdB, -rango_dB, 0.0), (-rango_dB, 0.0)
    else:
        return I, (float(I.min()), float(I.max()))

Uz0, dxz0 = fresnel_1fft(U0, dx0, lam, z0)
I0 = np.abs(Uz0)**2
extent0 = [-(N//2)*dxz0*1e3, (N//2)*dxz0*1e3, -(N//2)*dxz0*1e3, (N//2)*dxz0*1e3]
Ishow0, clim0 = preparar_imagen(I0)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.22)
im = ax.imshow(Ishow0, extent=extent0)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"Fresnel (1-FFT) — Intensidad {'(dB)' if usar_log else '|U|^2'}  z = {z0*1e3:.2f} mm")
im.set_clim(*clim0)
cbar = fig.colorbar(im, ax=ax, label=("dB" if usar_log else "Intensidad (u.a.)"))

# Zoom visual (solo reencuadre)
medio_mm = (N//2)*dxz0*1e3
ax.set_xlim(-medio_mm/zoom_vista, medio_mm/zoom_vista)
ax.set_ylim(-medio_mm/zoom_vista, medio_mm/zoom_vista)

# Slider
ax_z = plt.axes([0.12, 0.08, 0.76, 0.04])
z_slider = Slider(ax=ax_z, label="z (mm)",
                  valmin=z_min_m*1e3, valmax=z_max_slider_m*1e3, valinit=z0*1e3)

def on_z_change(_):
    z_m = z_slider.val * 1e-3
    Uz, dxz = fresnel_1fft(U0, dx0, lam, z_m)
    I = np.abs(Uz)**2
    Ishow, clim = preparar_imagen(I)
    # actualizar datos y escala
    im.set_data(Ishow)
    im.set_clim(*clim)
    # actualizar extent porque Δx_z cambia con z
    extent = [-(N//2)*dxz*1e3, (N//2)*dxz*1e3, -(N//2)*dxz*1e3, (N//2)*dxz*1e3]
    im.set_extent(extent)
    # mantener zoom relativo al nuevo tamaño físico
    medio = (N//2)*dxz*1e3
    ax.set_xlim(-medio/zoom_vista, medio/zoom_vista)
    ax.set_ylim(-medio/zoom_vista, medio/zoom_vista)
    ax.set_title(f"Fresnel (1-FFT) — Intensidad {'(dB)' if usar_log else '|U|^2'}  z = {z_m*1e3:.2f} mm")
    fig.canvas.draw_idle()

z_slider.on_changed(on_z_change)
plt.show()

# ----------------------- Resumen por consola -----------------------------
print("=== Resumen ===")
print(f"N={N}, dx0={dx0:.3e} m, λ={lam:.3e} m, lado={lado_abertura_m:.3e} m")
print(f"z_min (Fresnel 1-FFT) = (N/2)·dx0^2/λ = {z_min_m:.3e} m  ({z_min_m*1e3:.2f} mm)")
print(f"Slider: z ∈ [{z_min_m*1e3:.2f}, {z_max_slider_m*1e3:.2f}] mm")
