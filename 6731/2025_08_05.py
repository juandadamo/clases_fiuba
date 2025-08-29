import numpy as np
import matplotlib.pyplot as plt
import ht
import CoolProp as cp  


G_leche = 250 #lt/hora
T_vaca = 38.6 #C
T_ingreso_leche = 13 #C
T_agua_pozo = 10 #C
G_agua = 0.72 #m3/hora

rho_leche = 1030 #kg/m3
Cp_leche = 3.9 #kJ/kg.K
rho_agua = 1000 #kg/m3
Cp_agua = 4.18 #kJ/kg.K 

G_leche_kg_s = G_leche / 3600 * rho_leche/1000 #kg/s
G_agua_kg_s = G_agua * rho_agua / 3600 

Q_intercambio = G_leche_kg_s * Cp_leche * (T_vaca - T_ingreso_leche) #kW
T_agua_salida = T_agua_pozo + Q_intercambio / (G_agua_kg_s * Cp_agua) #C

C_leche = Cp_leche * G_leche_kg_s #kW/K
C_agua = Cp_agua * G_agua_kg_s #kW/K

Cmin = min(C_leche, C_agua) #kW/K
Cmax = max(C_leche, C_agua) #kW/K
Cr = Cmin / Cmax #-
delta_Tmax = T_vaca - T_agua_pozo #C
Qmax = Cmin * delta_Tmax #kW
eficiencia = Q_intercambio / Qmax #-
print(f"Q_intercambio: {Q_intercambio:.2f} kW")
print(f"T_agua_salida: {T_agua_salida:.2f} C")
print(f"Cmin: {Cmin:.2f} kW/K")
print(f"Cmax: {Cmax:.2f} kW/K")
print(f"Cr: {Cr:.2f}")
print(f"delta_Tmax: {delta_Tmax:.2f} C")
print(f"Qmax: {Qmax:.2f} kW")
print(f"Eficiencia: {eficiencia:.2f}")


Nut = ht.NTU_from_effectiveness(eficiencia, Cr,subtype='counterflow')

print(f"NTU: {Nut:.2f}")
U0 = 1 # kW/m2.K
Area = Nut * Cmin / U0 # m2
D_ext = 50e-3 # m
Longitud = Area / (np.pi * D_ext) # m
print(f"Area: {Area:.2f} m2")   
print(f"Longitud: {Longitud:.2f} m")

G_agua_kg_s_2 = G_agua_kg_s * 2 # kg/s
C_agua_2 = Cp_agua * G_agua_kg_s_2 # kW/K

Cmin_2 = min(C_leche, C_agua_2) #kW/K
Cmax_2 = max(C_leche, C_agua_2) #kW/K
Cr_2 = Cmin_2/ Cmax_2
Nut_2 = U0*Area / Cmin_2
eficiencia_2 = ht.effectiveness_from_NTU(Nut_2, Cr_2, subtype='counterflow')
print(f"NTU_2: {Nut_2:.2f}")
print(f"Eficiencia_2: {eficiencia_2:.2f}")
Q_intercambiado_2 = Qmax * eficiencia_2
T_ingreso_leche_2 = T_vaca - Q_intercambiado_2 / C_leche

print(f"T_ingreso_leche_2: {T_ingreso_leche_2:.2f} C")




 