# ---------------------------------------------------------------
# ASM (Angular Spectrum Method) — Abertura cuadrada con slider de z
# Pasos: 1) U(x,y;0) -> 2) FFT centrada -> 3) *H(z)* -> 4) IFFT centrada
# El "shift" está embebido en las funciones fft2c/ifft2c.
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------- FFT centrada y utilidades --------------------
def fft2c(u: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))

def ifft2c(U: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

def frequency_mesh(N: int, dx: float):
    f1d = np.fft.fftshift(np.fft.fftfreq(N, d=dx))  # ciclos/metro
    FX, FY = np.meshgrid(f1d, f1d, indexing="xy")
    return FX, FY

def make_square_aperture(N: int, dx: float, width: float) -> np.ndarray:
    x = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    half = width/2
    return ((np.abs(X) <= half) & (np.abs(Y) <= half)).astype(np.complex128)

def pad_to(u: np.ndarray, Np: int) -> np.ndarray:
    N = u.shape[0]
    pad = Np - N
    pre, post = pad//2, pad - pad//2
    return np.pad(u, ((pre, post), (pre, post)), mode="constant")

def crop_center(a: np.ndarray, N: int) -> np.ndarray:
    Ny, Nx = a.shape
    sy, sx = (Ny-N)//2, (Nx-N)//2
    return a[sy:sy+N, sx:sx+N]

def z_limit(N: int, dx: float, wavelength: float) -> float:
    # z_max = (M * dx^2) / lambda, con M = N/2
    return (N//2) * (dx**2) / wavelength

# --------------------------- Parámetros ---------------------------
wavelength     = 532e-9      # λ [m]
N              = 512         # muestras por eje (vista NxN)
dx             = 5e-6        # paso espacial Δx [m/píxel]
aperture_width = 200e-6      # lado del cuadrado [m]
pad_factor     = 2           # 1=sin padding, >=2 reduce wrap-around

# ---------------------- Campo de entrada U0 -----------------------
u0 = make_square_aperture(N, dx, aperture_width)
if pad_factor > 1:
    Np = int(2**np.ceil(np.log2(N * pad_factor)))  # potencia de 2 cercana
    u0_big = pad_to(u0, Np)
    dx_big = dx
else:
    Np = N
    u0_big = u0
    dx_big = dx

# Pre-cálculos para que el slider sea rápido
FX, FY = frequency_mesh(Np, dx_big)
U0 = fft2c(u0_big)
kz = 2*np.pi * np.sqrt((1.0/wavelength**2) - (FX**2 + FY**2) + 0j)  # incluye evanescentes

# Rango de z (según tu lámina)
z_max = z_limit(N, dx, wavelength)   # usa N y dx de la vista
z0 = 0.05 * z_max                     # valor inicial (>0)

# --------------------------- Interfaz -----------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.20)

extent_half = (N//2)*dx
H0 = np.exp(1j * z0 * kz)
Uz0_big = ifft2c(U0 * H0)
Uz0 = crop_center(Uz0_big, N) if (pad_factor > 1) else Uz0_big
im = ax.imshow(np.abs(Uz0)**2,
               extent=[-extent_half*1e3, extent_half*1e3,
                       -extent_half*1e3, extent_half*1e3])
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"Intensidad |U|^2  —  z = {z0*1e3:.3f} mm")
cbar = fig.colorbar(im, ax=ax, label="Intensidad (u.a.)")

# Slider (en mm para el usuario)
ax_z = plt.axes([0.12, 0.07, 0.76, 0.04])
z_slider = Slider(ax=ax_z, label="z (mm)", valmin=0.0, valmax=z_max*1e3, valinit=z0*1e3)

def update(_):
    z_m = z_slider.val * 1e-3  # mm -> m
    H = np.exp(1j * z_m * kz)
    Uz_big = ifft2c(U0 * H)
    Uz = crop_center(Uz_big, N) if (pad_factor > 1) else Uz_big
    I = np.abs(Uz)**2
    im.set_data(I)
    im.set_clim(vmin=float(I.min()), vmax=float(I.max()))  # auto-escala
    ax.set_title(f"Intensidad |U|^2  —  z = {z_m*1e3:.3f} mm")
    fig.canvas.draw_idle()

z_slider.on_changed(update)

print("=== Parámetros ===")
print(f"N={N}, dx={dx:.3e} m, λ={wavelength:.3e} m, lado={aperture_width:.3e} m")
print(f"Δf = 1/(N·dx) = {1.0/(N*dx):.3e} ciclos/m,  M=N/2={N//2}")
print(f"z_max = M·dx²/λ = {z_max:.3e} m  ({z_max*1e3:.3f} mm)")

plt.show()
