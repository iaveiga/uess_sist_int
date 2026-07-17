import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

class AlmacenEnv(gym.Env):
    def __init__(self, mapa, inventario, catalogo_skus, sku_objetivo, max_pasos=200):
        super(AlmacenEnv, self).__init__()
        
        self.mapa = mapa
        self.inventario_original = copy.deepcopy(inventario) 
        self.inventario = copy.deepcopy(inventario)
        
        self.alto, self.ancho = mapa.shape
        self.catalogo_skus = list(catalogo_skus)
        self.num_skus = len(self.catalogo_skus)
        self.sku_objetivo = sku_objetivo
        self.id_objetivo = self.catalogo_skus.index(sku_objetivo)
        
        # Este límite evita bucles infinitos durante el entrenamiento
        self.max_pasos = max_pasos
        self.pasos_actuales = 0
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.ancho),
            spaces.Discrete(self.alto),
            spaces.Discrete(4),
            spaces.Discrete(2),
            spaces.Discrete(self.num_skus)
        ))
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_x = 0
        self.robot_y = 0
        self.theta = 0  # 0: Norte, 1: Este, 2: Sur, 3: Oeste
        self.tiene_producto = 0 
        self.pasos_actuales = 0
        
        self.inventario = copy.deepcopy(self.inventario_original)
        
        estado_inicial = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        return estado_inicial, {}

    def step(self, action):
        self.pasos_actuales += 1
        prev_x, prev_y = self.robot_x, self.robot_y
        
        # Penalización base por paso (incentiva la velocidad y rutas cortas)
        recompensa = -1
        terminado = False
        truncado = False
        
        # --- 1. EJECUCIÓN DE ACCIONES ---
        
        if action == 0:  # Avanzar
            if self.theta == 0:    # Norte
                self.robot_y = max(0, self.robot_y - 1)
            elif self.theta == 1:  # Este
                self.robot_x = min(self.ancho - 1, self.robot_x + 1)
            elif self.theta == 2:  # Sur
                self.robot_y = min(self.alto - 1, self.robot_y + 1)
            elif self.theta == 3:  # Oeste
                self.robot_x = max(0, self.robot_x - 1)
                
            # Verifica si el robot chocó con una percha o no se mueve por estar frente a una pared externa
            if self.mapa[self.robot_y, self.robot_x] == 1:
                recompensa = -15  # Penalización por chocar contra percha
                self.robot_x, self.robot_y = prev_x, prev_y  # Deshacer movimiento
            elif self.robot_x == prev_x and self.robot_y == prev_y:
                recompensa = -10  # Penalización por chocar contra límites exteriores del mapa
                
        elif action == 1:  # Girar Izquierda
            self.theta = (self.theta - 1) % 4
            recompensa = -2   # Castigo ligeramente mayor para evitar giros infinitos sin sentido
            
        elif action == 2:  # Girar Derecha
            self.theta = (self.theta + 1) % 4
            recompensa = -2   # Castigo ligeramente mayor para evitar giros infinitos sin sentido
            
        elif action == 3:  # Extrae el objeto
            encontrado = False
            dx, dy = 0, 0
            if self.theta == 0: dy = -1
            elif self.theta == 1: dx = 1
            elif self.theta == 2: dy = 1
            elif self.theta == 3: dx = -1
            
            px, py = self.robot_x + dx, self.robot_y + dy
            
            # Verificar si enfrente hay una percha válida
            if 0 <= px < self.ancho and 0 <= py < self.alto and self.mapa[py, px] == 1:
                percha = self.inventario.get((px, py), {})
                for nivel in percha:
                    if self.sku_objetivo in percha[nivel]:
                        idx = percha[nivel].index(self.sku_objetivo)
                        percha[nivel][idx] = None  # Retirar objeto
                        self.tiene_producto = 1
                        recompensa = 150  # Recompensa por recolectar el objetivo
                        encontrado = True
                        break
                
            if not encontrado:
                recompensa = -20  # Se castiga por perder el tiempo intentado extraer donde no hay


                
        if self.robot_x == 0 and self.robot_y == 0 and self.tiene_producto == 1:
            recompensa = 300  # Recompensa considerable por completar la misión con éxito
            terminado = True

        
        if self.pasos_actuales >= self.max_pasos:
            truncado = True

        info = {}
        estado = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        
        return estado, recompensa, terminado, truncado, info
