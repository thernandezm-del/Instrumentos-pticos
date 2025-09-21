import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# CONFIGURACIÓN — modifica aquí tus parámetros principales
# ======================================================================
z_max_m  = 200
N                   = 2048        # [muestras] por eje (malla cuadrada N×N)
dx                  = 13e-6       # [m/píxel] paso espacial en el plano de la abertura
longitud_onda_m     = 532e-9     # [m] λ (ej. 544 nm; usa 532e-9 si es 532 nm)
freq=1/longitud_onda_m
lado_abertura_m     = 300e-6     # [m] lado del cuadrado de la abertura
centro_abertura_m   = (0.0, 0.0) # [m] (x0, y0) del centro de la abertura
amplitud_interior   = 1.0 + 0.0j # valor COMPLEJO dentro de la abertura (puede llevar fase)
fraccion_z_de_zmax  = 0.4   # z = fracción * z_max (guía práctica)
mostrar_abertura    = True       # dibujar la abertura (z=0) además de la intensidad en z
# ======================================================================

# ----------------------------------------------------------------------
# PARTE 1 — Malla espacial (x, y) y rejilla 2D (X, Y)
# Convenciones:
#   - x, y: vectores 1D (metros), centrados en 0
#   - X, Y: mallas 2D (metros), para evaluar U(x,y) en la rejilla
# ----------------------------------------------------------------------
Lx = N * dx   # [m] tamaño físico en x (solo informativo)
Ly = N * dx   # [m] tamaño físico en y (cuadrada)

# Vectores 1D (centrados en 0). Para N PAR, 0 cae en el índice N//2.
x = (np.arange(N) - N//2) * dx   # [m]
y = (np.arange(N) - N//2) * dx   # [m]

# Mallas 2D
X, Y = np.meshgrid(x, y, indexing="xy")  # shape (N, N)

# ----------------------------------------------------------------------
# PARTE 2 — Malla de frecuencias (fx, fy) y rejilla 2D (FX, FY)
# Hechos de muestreo:
#   Δf = 1/(N·dx)       (resolución en frecuencia)
#   f_max = 1/(2·dx)    (Nyquist)
# Usamos fftshift para centrar f=0.
# ----------------------------------------------------------------------
f1d_cpm = np.fft.fftshift(np.fft.fftfreq(N, d=dx))  # [ciclos/metro] centrado
FX_cpm, FY_cpm = np.meshgrid(f1d_cpm, f1d_cpm, indexing="xy")

# (informativo)
delta_f_cpm = 1.0/(N*dx)
fny_cpm     = 1.0/(2.0*dx)

# ----------------------------------------------------------------------
# Abertura cuadrada: máscara binaria COMPLEJA en la malla (X, Y)
# ----------------------------------------------------------------------
def mascara_abertura_cuadrada(X: np.ndarray,
                              Y: np.ndarray,
                              lado_m: float,
                              centro_m: tuple[float, float] = (0.0, 0.0),
                              amplitud_compleja: complex = 1.0 + 0.0j) -> np.ndarray:
    """
    Devuelve U0(x,y) COMPLEJO: amplitud_compleja dentro del cuadrado, 0 fuera.
    X, Y y 'lado_m' en metros. 'centro_m' permite desplazar la abertura.
    """
    x0_m, y0_m = centro_m
    semi = lado_m/2.0
    dentro = (np.abs(X - x0_m) <= semi) & (np.abs(Y - y0_m) <= semi)
    U0 = np.zeros_like(X, dtype=np.complex128)
    U0[dentro] = amplitud_compleja
    return U0

U0 = mascara_abertura_cuadrada(X, Y, lado_m=lado_abertura_m,
                               centro_m=centro_abertura_m,
                               amplitud_compleja=amplitud_interior)

# ----------------------------------------------------------------------
# PARTE 3 — FFT centrada (helpers)
#   U(x,y)  -> A(fx,fy) : fft2_centrada
#   A(fx,fy)-> U(x,y)   : ifft2_centrada
# ----------------------------------------------------------------------
def fft2_centrada(u_xy: np.ndarray) -> np.ndarray:
    """U(x,y) -> A(fx,fy) con DC en el centro."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_xy)))

def ifft2_centrada(A_f: np.ndarray) -> np.ndarray:
    """A(fx,fy) -> U(x,y) con DC en el centro."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A_f)))

A0 = fft2_centrada(U0)  # espectro en z=0

# ----------------------------------------------------------------------
# PARTE 4 — Propagador ASM y propagación a z fijo
#   k_z = 2π * sqrt( 1/λ^2 - (fx^2 + fy^2) )
#   H   = exp( i * z * k_z )
#   U_z = IFFT{ A0 * H }
# ----------------------------------------------------------------------
def calcular_kz_radpm(FX_cpm: np.ndarray,
                      FY_cpm: np.ndarray,
                      longitud_onda_m: float) -> np.ndarray:
    """
    k_z(fx,fy) en [rad/m], con FX,FY en [ciclos/m]. Incluye modos evanescentes.
    """
    return 2.0*np.pi * np.sqrt((1.0/longitud_onda_m**2) - (FX_cpm**2 + FY_cpm**2) + 0j)

def transfer_function_ASM(kz_radpm: np.ndarray, z_m: float) -> np.ndarray:
    """H(fx,fy; z) = exp(i * z * k_z)."""
    return np.exp(1j * z_m * kz_radpm)

kz_radpm = calcular_kz_radpm(FX_cpm, FY_cpm, longitud_onda_m)
z_m      = float(fraccion_z_de_zmax) * z_max_m
print(z_max_m)
H = transfer_function_ASM(kz_radpm, z_m)
U_z = ifft2_centrada(A0 * H)
I_z = np.abs(U_z)**2

# ----------------------------------------------------------------------
# PARTE 5 — Gráficas
# ----------------------------------------------------------------------
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]

if mostrar_abertura:
    plt.figure()
    plt.title("Abertura cuadrada — amplitud (z = 0)")
    plt.imshow(np.abs(U0), extent=extent_mm)
    plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
    plt.colorbar(label="Amplitud"); plt.tight_layout()

plt.figure()
plt.title(f"ASM — Intensidad |U(x,y;z)|^2   z = {z_m*1e3:.2f} mm")
plt.imshow(I_z, extent=extent_mm)
plt.xlabel("x (mm)"); plt.ylabel("y (mm)")
plt.colorbar(label="Intensidad (u.a.)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# PARTE 6 — Resumen por consola (útil para verificar muestreo)
# ----------------------------------------------------------------------
print("=== Resumen ===")
print(f"N = {N}, dx = {dx:.3e} m  ->  Lx = Ly = {Lx:.3e} m")
print(f"λ = {longitud_onda_m:.3e} m, lado = {lado_abertura_m:.3e} m, centro = {centro_abertura_m}")
print(f"Δf = {delta_f_cpm:.6e} ciclos/m  |  f_Nyquist = {fny_cpm:.6e} ciclos/m")
print(f"z_max ≈ (N/2)·dx²/λ = {z_max_m:.3e} m  ({z_max_m*1e3:.3f} mm)")
print(f"z = {z_m:.3e} m  ({z_m*1e3:.3f} mm)  = {fraccion_z_de_zmax*100:.1f}% de z_max")
