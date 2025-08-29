# -*- coding: utf-8 -*-
import numpy as np
import ht
import CoolProp as cp
import fluids


class fluido_intercambiador:    
    def __init__(self, nombre_fluido):
        self.fluid = nombre_fluido
        self.k,self.rho,self.Pr,self.mu,self.nu,self.cp = np.ones((6,1))
        self._temp_entrada = 20
        self._temp_salida = 30
        self.caudal = 1
        self.area = 1
        self.long = 1
        self._calor_intercambiado = 1
        self.propiedades()
        self.flowrate()
        self.Nu = 1
        self.hc = 1
        self.DeltaP = 1

    @property
    def temp_entrada(self):
        return self._temp_entrada
    @temp_entrada.setter
    def temp_entrada(self, value):
        self._temp_entrada = value
        # operaciones que quieras que se hagan
        self.propiedades()
    @property
    def temp_salida(self):
        return self._temp_salida
    @temp_salida.setter
    def temp_salida(self, value):
        self._temp_salida = value
        # operaciones que quieras que se hagan
        self.propiedades()
    @property
    def calor_intercambiado(self):
        return self._calor_intercambiado
    @calor_intercambiado.setter
    def calor_intercambiado(self,value):
        self._calor_intercambiado = value
        # caudal
        self.flowrate()
     
            
        
    def propiedades(self):
        self.temp_media = 1
        self.temp_media = (self.temp_salida+self.temp_entrada) / 2
        self.cp = cp.CoolProp.PropsSI('C','T',273+self.temp_media,'P',101.325e3,'Air')
        self.rho = cp.CoolProp.PropsSI('D','T',273+self.temp_media,'P',101.325e3,'Air')
        self.mu = cp.CoolProp.PropsSI('V','T',273+self.temp_media,'P',101.325e3,'Air')
        self.Pr = cp.CoolProp.PropsSI('Prandtl','T',273+self.temp_media,'P',101.325e3,'Air')
        self.k = cp.CoolProp.PropsSI('L','T',273+self.temp_media,'P',101.325e3,'Air')
        self.nu = self.mu/self.rho

    def flowrate(self):
        self.caudal = np.abs(self.calor_intercambiado/self.rho/(self.cp*(self.temp_entrada-self.temp_salida)))
        self.veloc  = self.caudal  / self.area
    
    def Reynolds(self):
        self.Re = self.veloc * self.long / self.nu
    
        
