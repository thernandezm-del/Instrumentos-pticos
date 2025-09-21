# ---------------------------------------------------------------
# ASM — Abertura cuadrada con slider de z y slider de ZOOM
# - z: controla el crecimiento físico del patrón (~ lambda*z/a)
# - zoom: recorte central [5%..100%] del campo mostrado (amplía en pantalla)
# - a_um: (opcional) cambia el lado de la abertura para ver su efecto
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------- FFT centrada y utilidades --------------------
def fft2c(u):  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
def ifft2c(U): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

def freq_mesh(N, dx):
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dx))  # ciclos/m
    FX, FY = np.meshgrid(f, f, indexing="xy")
    return FX, FY

def square_aperture(N, dx, width):
    x = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    h = width/2
    return ((np.abs(X) <= h) & (np.abs(Y) <= h)).astype(np.complex128)

def pad_center(u, Np):
    N = u.shape[0]
    pad = Np - N
    a, b = pad//2, pad - pad//2
    return np.pad(u, ((a, b), (a, b)), mode="constant")

def crop_center(a, N):
    Ny, Nx = a.shape
    sy, sx = (Ny-N)//2, (Nx-N)//2
    return a[sy:sy+N, sx:sx+N]

def z_limit(N, dx, lam):     # guía práctica de tus láminas
    return (N//2) * dx*dx / lam

# --------------------------- Parámetros ---------------------------
lam = 532e-9        # longitud de onda [m]
N   = 2048          # muestras de la "vista" (figura) N x N
dx  = 10e-6          # paso espacial [m/píxel]
a   = 200e-6        # lado de la abertura [m]
pad_factor = 2      # 1=sin padding; >=2 reduce wrap-around

# ---------------------- Construcción inicial ----------------------
# Campo de entrada + padding para cálculo (mejorar bordes FFT)
u0 = square_aperture(N, dx, a)
if pad_factor > 1:
    Np = int(2**np.ceil(np.log2(N*pad_factor)))    # potencia de 2 cercana
    u0b = pad_center(u0, Np)
    dxb = dx
else:
    Np  = N
    u0b = u0
    dxb = dx

FX, FY = freq_mesh(Np, dxb)
U0b = fft2c(u0b)
kz = 2*np.pi * np.sqrt((1.0/lam**2) - (FX**2 + FY**2) + 0j)  # ASM exacto

# Rango de z y valores iniciales
zmax = z_limit(N, dx, lam)   # límite de aliasing útil
z0   = 0.5 * zmax            # arranque en el medio del rango

# --------------------------- Interfaz ------------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.30)  # espacio para 2 sliders

# Propagación inicial
def propagate_and_show(z, zoom_frac, a_current):
    """
    z: distancia en metros
    zoom_frac: fracción del campo final a mostrar (0.05..1.0)
    a_current: lado abertura [m] (permite cambiarla en vivo)
    """
    # Si cambia el tamaño de la abertura, rehacemos U0b y U0 espectral una sola vez:
    global u0b, U0b, Np, pad_factor, dxb
    # Regenerar si 'a' cambió respecto al usado para U0b
    if not np.isclose(a_current, propagate_and_show._a_used):
        u0_new = square_aperture(N, dx, a_current)
        if pad_factor > 1:
            u0b = pad_center(u0_new, Np)
        else:
            u0b = u0_new
        U0b = fft2c(u0b)
        propagate_and_show._a_used = a_current

    # Propaga ASM
    Uz_big = ifft2c(U0b * np.exp(1j * z * kz))
    Uz     = crop_center(Uz_big, N) if (pad_factor > 1) else Uz_big
    I      = np.abs(Uz)**2

    # ---- ZOOM (recorte central) ----
    zoom_frac = np.clip(zoom_frac, 0.05, 1.0)  # 5%..100% del tamaño original
    Nv = max(8, int(N * zoom_frac))           # tamaño de la ventana de vista
    # garantizar impar para centrar exacto (opcional)
    if Nv % 2 == 0:
        Nv -= 1
    Iz = crop_center(I, Nv)

    # Extensión física del recorte para los ejes (en mm)
    half = (Nv//2) * dx
    extent = [-half*1e3, half*1e3, -half*1e3, half*1e3]   # mm

    return Iz, extent

# estado inicial del tamaño de abertura usado
propagate_and_show._a_used = a

I0, extent0 = propagate_and_show(z0, zoom_frac=1.0, a_current=a)
im = ax.imshow(I0, extent=extent0)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"ASM |U|^2  —  z = {z0*1e3:.3f} mm  |  zoom = 100%  |  a = {a*1e6:.0f} µm")
cbar = fig.colorbar(im, ax=ax, label="Intensidad (u.a.)")

# Slider de z (mm)
ax_z = plt.axes([0.12, 0.16, 0.76, 0.05])
s_z = Slider(ax=ax_z, label="z (mm)", valmin=0.0, valmax=zmax*1e3, valinit=z0*1e3)

# Slider de ZOOM (% del tamaño original mostrado)
ax_zoom = plt.axes([0.12, 0.09, 0.76, 0.05])
s_zoom = Slider(ax=ax_zoom, label="zoom (%)", valmin=5, valmax=100, valinit=100)

# (Opcional) Slider de abertura (µm) para ver su efecto en el tamaño del patrón
ax_a = plt.axes([0.12, 0.02, 0.76, 0.05])
s_a = Slider(ax=ax_a, label="lado a (µm)", valmin=50, valmax=400, valinit=a*1e6)

def update(_):
    z_m = s_z.val * 1e-3                 # mm -> m
    zoom_frac = s_zoom.val / 100.0       # %
    a_cur = s_a.val * 1e-6               # µm -> m
    I, ext = propagate_and_show(z_m, zoom_frac, a_cur)
    im.set_data(I)
    im.set_extent(ext)
    im.set_clim(vmin=float(I.min()), vmax=float(I.max()))  # auto-escala
    ax.set_title(f"ASM |U|^2  —  z = {z_m*1e3:.3f} mm  |  zoom = {int(s_zoom.val)}%  |  a = {a_cur*1e6:.0f} µm")
    fig.canvas.draw_idle()

s_z.on_changed(update)
s_zoom.on_changed(update)
s_a.on_changed(update)

print("=== Parámetros ===")
print(f"N={N}, dx={dx:.3e} m, λ={lam:.3e} m, a={a:.3e} m, pad_factor={pad_factor}")
print(f"z_max ≈ (N/2)·dx²/λ = {zmax:.3e} m  ({zmax*1e3:.3f} mm)")
print("Usa 'z' para crecer físicamente el patrón; 'zoom' solo amplía la vista.")

plt.show()
