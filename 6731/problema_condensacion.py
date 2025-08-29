import numpy as np 
import CoolProp.CoolProp as cp
import ht
import matplotlib.pyplot as plt
T_w = 273+70 # Temperatura de tubos
T_agua_salida = T_w-65
Tmedia = 40+273 #primera suposicion

T_sat = cp.PropsSI('T', 'P', 101325, 'Q', 0, 'Water') # Temperatura de saturación del agua a presión atmosférica en K
m_agua_refrigerante = 70 # Masa de refrigerante en kg/s
n_tubos = 90 # Número de tubos  
D_tubo_interior = 2e-2 # Diámetro interior del tubo en m
D_tubo_exterior = 2.5e-2 # Diámetro exterior del tubo en m
L_tubo = 2 # Longitud del tubo en m
T_agua_refrigerante = 273+50 # Temperatura del agua refrigerante en K
P_saturacion = cp.PropsSI('P', 'T', Tmedia, 'Q', 0, 'Water') # Presión de saturación del agua a la temperatura de condensación en Pa

Area_interior = np.pi * (D_tubo_interior / 2)**2 * n_tubos * L_tubo # Área interior de los tubos en m²
Area_exterior = np.pi * (D_tubo_exterior / 2)**2 * n_tubos * L_tubo # Área exterior de los tubos en m²
# Nu_condensacion = ht.Nusselt_Perry(D_tubo_interior, D_tubo_exterior, L_tubo, T_agua_condensacion, m_agua_refrigerante, 'Water', 'Water', 0.5) # Número de Nusselt para condensación

Cp_agua = cp.PropsSI('C', 'T', Tmedia, 'Q', 0, 'Water') # Calor específico del agua en J/(kg·K)
Cmin = m_agua_refrigerante * Cp_agua # Capacidad térmica mínima en W/K
rho_l = cp.PropsSI('D', 'T', Tmedia, 'Q', 0, 'Water') # Densidad del líquido en kg/m³
rho_v = cp.PropsSI('D', 'T', T_sat, 'Q', 1, 'Water') # Densidad del vapor en kg/m³
k_l = cp.PropsSI('L', 'T', Tmedia, 'Q', 0, 'Water') # Conductividad térmica del líquido en W/(m·K)
mu_l = cp.PropsSI('V', 'T', Tmedia, 'Q', 0, 'Water') # Viscosidad dinámica del líquido en Pa·s
hfg = cp.PropsSI('H', 'T', T_sat, 'Q', 1, 'Water') - cp.PropsSI('H', 'T', T_sat, 'Q', 0, 'Water') # Calor de vaporización en J/kg
Perimetro_tubo_ext = np.pi * D_tubo_exterior  
Nu_cond = ht.condensation.Nusselt_laminar(T_sat,T_w,rho_v,rho_l,k_l,mu_l,hfg,Perimetro_tubo_ext)
h_cond = Nu_cond * k_l / D_tubo_exterior # Coeficiente de transferencia de calor por condensación en W/(m²·K)
Q_cond = h_cond * Area_exterior * (T_sat- T_w) * n_tubos

masa_cond = Q_cond / hfg # Masa de vapor condensado en kg/s

Q_conveccion = Q_cond 


T_agua_entrada = T_agua_salida - Q_conveccion / Cmin  / (Area_interior*n_tubos)# Temperatura de entrada del agua refrigerante en K
dTmax = T_w - T_agua_entrada 
Qmax = Cmin * dTmax *Area_interior*n_tubos# Calor máximo que puede absorber el agua refrigerante en W
epsilon = Q_conveccion / Qmax # Eficiencia del sistema

Nut = ht.NTU_from_effectiveness(epsilon, 0)

UA = Cmin / Nut # Producto UA en W/K
U  = UA / (Area_interior*n_tubos) # Coeficiente global de transferencia de calor en W/(m²·K)