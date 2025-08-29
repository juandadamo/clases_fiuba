import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constantes
sigma = 5.670374419e-8  # Constante de Stefan-Boltzmann [W/m²K⁴]

# Parámetros del problema
D = 0.5  # Diámetro interno [m]
L = 2.0  # Longitud del tubo [m]
T_w = 800  # Temperatura de pared [K]
T_g = 1200  # Temperatura del gas [K]
epsilon_w = 0.85  # Emisividad de la pared
h = 50  # Coeficiente de convección [W/m²K]
P = 1.0  # Presión total [atm]
P_CO2 = 0.1  # Presión parcial CO2 [atm]
P_H2O = 0.08  # Presión parcial H2O [atm]
L_e = 0.9 * D  # Longitud media del haz [m]

# 1. Cálculo solo por convección
A = np.pi * D * L  # Área interna del tubo [m²]
Q_conv = h * A * (T_g - T_w)  # Transferencia de calor por convección [W]

print(f"Transferencia de calor solo por convección: {Q_conv/1000:.2f} kW")

# 2. Cálculo con radiación en gases (modelo Hottel y Sarofim)

# Datos para emisividad de CO2 y H2O (de tablas típicas)
# Valores aproximados basados en correlaciones de Hottel

# CO2: Temperatura [K] vs emisividad para P_CO2*L_e = 0.1*0.45 = 0.045 atm·m
T_CO2 = np.array([400, 600, 800, 1000, 1200, 1400, 1600])
epsilon_CO2 = np.array([0.09, 0.12, 0.15, 0.17, 0.19, 0.21, 0.22])

# H2O: Temperatura [K] vs emisividad para P_H2O*L_e = 0.08*0.45 = 0.036 atm·m
T_H2O = np.array([400, 600, 800, 1000, 1200, 1400, 1600])
epsilon_H2O = np.array([0.14, 0.20, 0.25, 0.29, 0.32, 0.34, 0.36])

# Factores de corrección por presión (C_CO2) y por sobreposición (Delta_epsilon)
# Para este ejemplo usamos valores típicos:
C_CO2 = 1.0  # Poco efecto a 1 atm
C_H2O = 1.1  # Corrección típica para H2O
Delta_epsilon = 0.05  # Factor de sobreposición típico

# Interpolación para obtener emisividades a T_g
epsilon_CO2_Tg = interp1d(T_CO2, epsilon_CO2, kind='linear')(T_g)
epsilon_H2O_Tg = interp1d(T_H2O, epsilon_H2O, kind='linear')(T_g)

# Emisividad total del gas
epsilon_g = C_CO2 * epsilon_CO2_Tg + C_H2O * epsilon_H2O_Tg - Delta_epsilon

# Factor de intercambio radiativo
F = 1 / (1/epsilon_w + 1/epsilon_g - 1)

# Transferencia de calor por radiación
Q_rad = sigma * A * F * (T_g**4 - T_w**4)

# Transferencia total (convección + radiación)
Q_total = Q_conv + Q_rad

print("\nIncluyendo radiación en gases:")
print(f"Emisividad CO2: {epsilon_CO2_Tg:.3f}")
print(f"Emisividad H2O: {epsilon_H2O_Tg:.3f}")
print(f"Emisividad total gas: {epsilon_g:.3f}")
print(f"Factor de intercambio: {F:.3f}")
print(f"Transferencia por radiación: {Q_rad/1000:.2f} kW")
print(f"Transferencia total: {Q_total/1000:.2f} kW")

# 3. Comparación
discrepancia = (Q_total - Q_conv) / Q_conv * 100

print(f"\nDiscrepancia: {discrepancia:.1f}%")

# Visualización de las curvas de emisividad
plt.figure(figsize=(10, 6))
plt.plot(T_CO2, epsilon_CO2, 'bo-', label='CO2 (P·L=0.045 atm·m)')
plt.plot(T_H2O, epsilon_H2O, 'rs-', label='H2O (P·L=0.036 atm·m)')
plt.xlabel('Temperatura (K)')
plt.ylabel('Emisividad')
plt.title('Emisividad de Gases (datos de referencia)')
plt.grid(True)
plt.legend()
plt.show()
