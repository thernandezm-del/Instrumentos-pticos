import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

# =================== CONFIGURACIÓN (según enunciado) ===================
E_NUM       = 3                       # <-- tu número de equipo: transm_E03.png
MASK_PATH   = f"transm_E{E_NUM:02d}.png"
Lx_mm       = 5.8                     # ancho físico de la transmitancia en x [mm]
lam         = 633e-9                  # longitud de onda [m] (633 nm)
Z_LIST_MM   = [130, 180, 210, 280, 330, 380]  # z a simular (mm)

BIN_THRESH  = 0.5                     # umbral de binarización (0..1)
NEGRO_TRANSMITE = True                # Negro=1 transmite, Blanco=0 bloquea
MOSTRAR_LOG = False                   # False => blanco/negro lineal; True => dB
RANGO_DB    = 40.0                    # si MOSTRAR_LOG=True
# =======================================================================

# ---------- utilidades ----------
def fft2c(u):  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
def ifft2c(U): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

def fresnel_1fft(U0, dx0, dy0, lam, z):
    """Fresnel 1-FFT (pre-chirp -> FFT*ΔxΔy -> post-chirp + prefactor)."""
    Ny, Nx = U0.shape
    k = 2*np.pi/lam

    x0 = (np.arange(Nx) - Nx//2) * dx0
    y0 = (np.arange(Ny) - Ny//2) * dy0
    X0, Y0 = np.meshgrid(x0, y0, indexing="xy")

    Uprime = U0 * np.exp(1j * k/(2*z) * (X0**2 + Y0**2))
    U2     = fft2c(Uprime) * (dx0 * dy0)

    dx_z = lam * z / (Nx * dx0)
    dy_z = lam * z / (Ny * dy0)
    xz = (np.arange(Nx) - Nx//2) * dx_z
    yz = (np.arange(Ny) - Ny//2) * dy_z
    Xz, Yz = np.meshgrid(xz, yz, indexing="xy")

    Uz = (np.exp(1j * k * z) / (1j * lam * z)) * np.exp(1j * k/(2*z) * (Xz**2 + Yz**2)) * U2
    return Uz, dx_z, dy_z

# ---------- carga y binarización ----------
raw = iio.imread(MASK_PATH)
if raw.ndim == 3:
    raw = 0.299*raw[...,0] + 0.587*raw[...,1] + 0.114*raw[...,2]  # a gris

g = raw.astype(float)
g = (g - g.min()) / (g.max() - g.min() + 1e-12)  # normaliza 0..1

# blanco=1, negro=0 por umbral; invertimos si negro debe transmitir
mask_blanco = (g >= BIN_THRESH).astype(float)  # 1=blanco, 0=negro
U0 = (1.0 - mask_blanco) if NEGRO_TRANSMITE else mask_blanco
U0 = U0.astype(np.complex128)  # onda plana unidad -> amplitud 1 dentro, 0 fuera

# ---------- malla física (5.8 mm en x; y se ajusta por aspecto) ----------
Ny, Nx = U0.shape
Lx = Lx_mm * 1e-3
dx0 = Lx / Nx
Ly = Lx * (Ny / Nx)           # conserva aspecto de la imagen
dy0 = Ly / Ny

# ---------- restricción de Fresnel ----------
dx_ref = min(dx0, dy0)
z_min = (max(Nx, Ny)//2) * (dx_ref**2) / lam
Z_LIST_M = []
for z_mm in Z_LIST_MM:
    z = max(z_min, z_mm*1e-3)          # impone z >= z_min
    Z_LIST_M.append(z)

print("=== Parámetros de la simulación ===")
print(f"Archivo: {MASK_PATH}  |  tamaño físico: {Lx_mm:.2f} mm (x)")
print(f"Resolución: {Nx}×{Ny} px  |  dx0={dx0*1e6:.3f} µm, dy0={dy0*1e6:.3f} µm")
print(f"λ = {lam*1e9:.1f} nm  |  z_min(Fresnel) ≈ {z_min*1e3:.2f} mm")
print("z (mm) solicitados -> usados (mm):")
print(", ".join([f"{z_req}→{(max(z_min, z_req*1e-3))*1e3:.1f}" for z_req in Z_LIST_MM]))

# ---------- cálculo y figura (B/N) ----------
nplots = len(Z_LIST_M)
nrows, ncols = (2, 3) if nplots <= 6 else (3, 3)
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 7), constrained_layout=True)
axes = np.array(axes).ravel()

for ax, z in zip(axes, Z_LIST_M):
    Uz, dxz, dyz = fresnel_1fft(U0, dx0, dy0, lam, z)
    I = np.abs(Uz)**2
    if MOSTRAR_LOG:
        In = I/(I.max()+1e-16); IdB = 10*np.log10(In+1e-16)
        img = np.clip(IdB, -RANGO_DB, 0.0)
        ax.imshow(img, cmap="gray", origin="upper", vmin=-RANGO_DB, vmax=0.0)
    else:
        img = I/(I.max()+1e-16)
        ax.imshow(img, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)

    ax.set_title(f"z = {z*1e3:.0f} mm", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# apaga ejes sobrantes
for ax in axes[nplots:]:
    ax.axis("off")

fig.suptitle(f"Transmittance: transm_E{E_NUM:02d}.png — Fresnel 1-FFT (λ=633 nm)", fontsize=12)
plt.show()
