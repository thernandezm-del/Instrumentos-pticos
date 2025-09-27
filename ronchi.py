import numpy as np
import matplotlib.pyplot as plt

# ===================== PARÁMETROS FÍSICOS ===================== #
lam = 633e-9                 # longitud de onda [m]
lpmm = 10                    # line pairs per mm
d = 1.0/(lpmm*1e3)           # periodo [m] => 1/(10 mm^-1) = 1e-4 m
duty = 0.5                   # Ronchi 50/50
zTalbot = 2*d*d/lam          # Talbot para rejilla de AMPLITUD

# ===================== MALLA ESPACIAL ========================= #
Lx = 2e-3                    # tamaño del campo simulado (x) [m]
Ly = 2e-3                    # tamaño del campo simulado (y) [m]
Nx = 1024                    # puntos en x  (ajusta si tu PC es lenta)
Ny = 1024                    # puntos en y
dx = Lx / Nx
dy = Ly / Ny

x = (np.arange(Nx) - Nx//2) * dx
y = (np.arange(Ny) - Ny//2) * dy
X, Y = np.meshgrid(x, y)

# ===================== REJILLA RONCHI (amplitud binaria) ===== #
# t(x) = 1 si cos(2π x/d) >= 0, 0 si < 0  (duty 0.5)
tx = 0.5*(1.0 + np.sign(np.cos(2*np.pi*x/d)))
t = np.tile(tx, (Ny, 1))        # constante en y (líneas paralelas al eje y)

# Campo incidente plano (amplitud 1) -> tras máscara:
U0 = t.astype(np.complex128)    # U0(x,y) = t(x,y)

# ===================== FRECUENCIAS ESPACIALES ================= #
fx = np.fft.fftfreq(Nx, d=dx)   # [ciclos/m]
fy = np.fft.fftfreq(Ny, d=dy)
FX, FY = np.meshgrid(fx, fy)

# ===================== PROPAGACIÓN (Fresnel, paraxial) ======= #
# Método del espectro angular (paraxial):
# A0 = F{U0};  H = exp(i 2π z / λ) * exp(-i π λ z (fx^2 + fy^2));
# Uz = F^{-1}{ A0 * H }
A0 = np.fft.fft2(U0)

def fresnel_propagate_fft(A0, FX, FY, lam, z):
    H = np.exp(1j*2*np.pi*z/lam) * np.exp(-1j*np.pi*lam*z*(FX**2 + FY**2))
    Uz = np.fft.ifft2(A0 * H)
    return Uz

# ===================== DISTANCIAS A EVALUAR =================== #
z_list = [0.0, zTalbot/4, zTalbot/2, zTalbot]

# ===================== SIMULACIÓN Y PLOTS ===================== #
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axs = axs.ravel()

corrs = []
line0 = np.abs(U0[Ny//2, :])**2   # línea central a z=0 (referencia binaria)

for k, z in enumerate(z_list):
    Uz = U0 if z==0 else fresnel_propagate_fft(A0, FX, FY, lam, z)
    Iz = np.abs(Uz)**2

    # Muestra una ventana central para que se vean bien las franjas
    frac = 0.5     # 50% del campo
    cx = int(Nx*(0.5-frac/2));  ex = int(Nx*(0.5+frac/2))
    cy = int(Ny*(0.5-frac/2));  ey = int(Ny*(0.5+frac/2))
    axs[k].imshow(Iz[cy:ey, cx:ex], cmap='gray', extent=[x[cx]*1e3, x[ex-1]*1e3, y[ey-1]*1e3, y[cy]*1e3])
    axs[k].set_title(f"Intensidad a z = {z*1e3:.2f} mm")
    axs[k].set_xlabel("x [mm]"); axs[k].set_ylabel("y [mm]")

    # Correlación 1D (línea central) vs patrón de referencia (z=0)
    linez = Iz[Ny//2, :]
    # normaliza y correlaciona sin lag (aprox para ver “parecido”)
    a = (linez - linez.mean()) / (linez.std() + 1e-12)
    b = (line0 - line0.mean()) / (line0.std() + 1e-12)
    corr = (a*b).mean()
    corrs.append(corr)

plt.suptitle(f"Ronchi {lpmm} lp/mm, λ = {lam*1e9:.0f} nm  |  z_T ≈ {zTalbot*1e3:.2f} mm", fontsize=12)
plt.show()

print(f"z_T (analítico) = {zTalbot*1e3:.3f} mm")
for z, c in zip(z_list, corrs):
    print(f"z = {z*1e3:7.3f} mm -> correlación línea central vs z=0: {c:+.3f}")
