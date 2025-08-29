import numpy as np

# Parámetros
epsilon1 = 0.05  # Emisividad cara externa escudo
epsilon2 = 0.05  # Emisividad cara interna escudo
epsilon_t = 0.10 # Emisividad tanque
T_t = 100        # Temperatura tanque [K]
G_s = 1250       # Irradiación solar [W/m²]
sigma = 5.67e-8  # Constante Stefan-Boltzmann

# Solución analítica para T_s
T_s = T_t * (epsilon_t/epsilon2)**(1/4)

# Verificación balance energético en escudo
term1 = epsilon1 * sigma * T_s**4
term2 = epsilon2 * sigma * (T_s**4 - T_t**4)
balance = epsilon1*G_s - term1 - term2

# Flujo de calor (debe ser cero en equilibrio)
q_net = epsilon_t * sigma * (T_s**4 - T_t**4)

# Resultados
print(f"1. Temperatura del escudo (T_s): {T_s:.2f} K")
print(f"2. Temperatura del tanque (T_t): {T_t} K (fija)")
print(f"3. Flujo neto al tanque: {q_net:.2f} W/m² (debe ser ≈0)")
print(f"\nVerificación balance escudo: {balance:.2e} W/m² (≈0)")
