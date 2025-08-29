import numpy as np
import matplotlib.pyplot as plt
import ht
import fluids
from CoolProp.CoolProp import PropsSI

# --- Datos ---
L = 1.0  # m, longitud caracteristica (cocodrilo)
D = 0.3  # m, ancho frontal
Tw = 30.0 + 273.15  # K, temperatura superficial
T_inf_15 = 15.0 + 273.15  # K, agua a 15 C
T_inf_20 = 20.0 + 273.15  # K, agua a 20 C

epsilon = 1e-3  # m, rugosidad de la piel

g = 9.81  # m/s^2, gravedad

# Rango de velocidades
U_inf = np.linspace(0.2, 2.0, 20)  # m/s

# --- Funciones auxiliares ---
def propiedades_agua(T):
    """Devuelve propiedades del agua a temperatura T (K)"""
    rho = PropsSI('D', 'T', T, 'P', 101325, 'Water')  # kg/m3
    mu = PropsSI('V', 'T', T, 'P', 101325, 'Water')   # Pa.s
    k = PropsSI('L', 'T', T, 'P', 101325, 'Water')    # W/m.K
    cp = PropsSI('C', 'T', T, 'P', 101325, 'Water')   # J/kg.K
    beta = 1 / T  # 1/K, coef. de expansion aproximado
    alpha = k / (rho * cp)  # difusividad termica
    nu = mu / rho  # viscosidad cinematica
    return rho, mu, k, cp, beta, alpha, nu

def calcular_Re(U, L, nu):
    return U * L / nu

def calcular_Gr(beta, deltaT, L, nu, alpha):
    return g * beta * deltaT * L**3 / (nu * alpha)

def calcular_Pr(nu, alpha):
    return nu / alpha

# --- Propiedades ---
props_15 = propiedades_agua(T_inf_15)
props_20 = propiedades_agua(T_inf_20)

# --- Cálculo de números adimensionales ---
Re_15 = calcular_Re(U_inf, L, props_15[-1])
Re_20 = calcular_Re(U_inf, L, props_20[-1])

Pr_15 = calcular_Pr(props_15[-1], props_15[-2])
Pr_20 = calcular_Pr(props_20[-1], props_20[-2])

# Delta T para convección natural
DeltaT_15 = Tw - T_inf_15
DeltaT_20 = Tw - T_inf_20

Ra_15 = calcular_Gr(props_15[4], DeltaT_15, L, props_15[-1], props_15[-2]) * Pr_15
Ra_20 = calcular_Gr(props_20[4], DeltaT_20, L, props_20[-1], props_20[-2]) * Pr_20

print(f"Prandtl a 15°C: {Pr_15:.2f}")
print(f"Prandtl a 20°C: {Pr_20:.2f}")
print(f"Rayleigh a 15°C: {Ra_15:.2e}")
print(f"Rayleigh a 20°C: {Ra_20:.2e}")

# --- Placeholder de correlaciones ---
# No se proveen expresamente: los estudiantes deben buscar

def h_conveccion_forzada(Re, Pr, k, L):
    # Placeholder tipo: Nusselt = 0.332 * Re**0.5 * Pr**(1/3)
    Nu = 0.332 * Re**0.5 * Pr**(1/3)
    return Nu * k / L

def h_conveccion_natural(Ra, k, L):
    # Placeholder tipo: Nu = C * Ra^n
    Nu = 0.54 * Ra**(1/4)  # para flujo laminar sobre superficie horizontal
    return Nu * k / L

# --- Graficos ---
h_15 = h_conveccion_forzada(Re_15, Pr_15, props_15[2], L)
h_20 = h_conveccion_forzada(Re_20, Pr_20, props_20[2], L)

plt.figure(figsize=(8,6))
plt.plot(U_inf, h_15, label="Agua a 15°C")
plt.plot(U_inf, h_20, label="Agua a 20°C")
plt.xlabel("Velocidad $U_\infty$ [m/s]")
plt.ylabel(r"Coeficiente de convección $\bar{h}$ [W/m$^2$K]")
plt.title("Convección Forzada sobre Cocodrilo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Conveccion natural (en reposo)
h_nat_15 = h_conveccion_natural(Ra_15, props_15[2], L)
h_nat_20 = h_conveccion_natural(Ra_20, props_20[2], L)

print(f"h promedio en reposo (15°C): {h_nat_15:.2f} W/m2K")
print(f"h promedio en reposo (20°C): {h_nat_20:.2f} W/m2K")

# --- Notas sobre rugosidad ---
# Se podría calcular un Re_critico basado en rugosidad relativa epsilon/L
# y analizar si el flujo es rugoso o hidraulicamente liso
Re_critico = 5e5 * (epsilon/L)**(-1/1.1)
print(f"Re_critico por rugosidad estimado: {Re_critico:.2e}")
