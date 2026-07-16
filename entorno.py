import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

class AlmacenEnv(gym.Env):
    def __init__(self, mapa, inventario, catalogo_skus, sku_objetivo):
        super(AlmacenEnv, self).__init__()
        
        self.mapa = mapa
        # Creamos una copia del inventario inicial
        # Puesto que operaremos y al ser objetos se modifican directametne
        self.inventario_original = copy.deepcopy(inventario) 
        self.inventario = copy.deepcopy(inventario)
        
        self.alto, self.ancho = mapa.shape
        self.catalogo_skus = list(catalogo_skus)
        self.num_skus = len(self.catalogo_skus)
        self.sku_objetivo = sku_objetivo
        self.id_objetivo = self.catalogo_skus.index(sku_objetivo)
        
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
        self.theta = 0 # Inicialmente mira al Norte (0)
        self.tiene_producto = 0 
        
        # Restauramos el inventario original
        self.inventario = copy.deepcopy(self.inventario_original)
        
        estado_inicial = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        return estado_inicial, {}

    def step(self, action):
        prev_x, prev_y = self.robot_x, self.robot_y
        recompensa = -1
        terminado = False
        
        # 1. Movimiento basado en Orientación (theta)
        # Acciones: 0 = Avanzar, 1 = Girar Izquierda, 2 = Girar Derecha
        if action == 0: # Avanzar en la dirección de theta
            if self.theta == 0:   # Norte
                self.robot_y = max(0, self.robot_y - 1)
            elif self.theta == 1: # Este
                self.robot_x = min(self.ancho - 1, self.robot_x + 1)
            elif self.theta == 2: # Sur
                self.robot_y = min(self.alto - 1, self.robot_y + 1)
            elif self.theta == 3: # Oeste
                self.robot_x = max(0, self.robot_x - 1)
                
        elif action == 1: # Girar Izquierda (antihorario)
            self.theta = (self.theta - 1) % 4
            
        elif action == 2: # Girar Derecha (horario)
            self.theta = (self.theta + 1) % 4
            
        # Penalización por chocar contra percha o pared
        if self.mapa[self.robot_y, self.robot_x] == 1:
            recompensa = -50
            self.robot_x, self.robot_y = prev_x, prev_y # Deshacer avance
            
        # 2. Acción Especial: Extraer (Acción 3)
        elif action == 3:
            encontrado = False
            # El robot intenta extraer el objeto que tiene justo enfrente según su orientación theta
            dx, dy = 0, 0
            if self.theta == 0: dy = -1   # Intenta extraer de la percha al Norte
            elif self.theta == 1: dx = 1  # Al Este
            elif self.theta == 2: dy = 1  # Al Sur
            elif self.theta == 3: dx = -1 # Al Oeste
            
            px, py = self.robot_x + dx, self.robot_y + dy
            
            # Verificar si enfrente hay una percha válida
            if 0 <= px < self.ancho and 0 <= py < self.alto and self.mapa[py, px] == 1:
                percha = self.inventario[(px, py)]
                for nivel in percha:
                    if self.sku_objetivo in percha[nivel]:
                        idx = percha[nivel].index(self.sku_objetivo)
                        percha[nivel][idx] = None  # Retirar de stock
                        self.tiene_producto = 1 # B_t = 1
                        recompensa = 100 # Recompensa por recolectar el producto correcto
                        encontrado = True
                        break
                
            if not encontrado:
                recompensa = -15 # Penalización por intentar extraer en vacío o pared
                
        # 3. Éxito: Volver a base con el producto
        if self.robot_x == 0 and self.robot_y == 0 and self.tiene_producto == 1:
            recompensa = 100
            terminado = True
            
        info = {}
        estado = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        return estado, recompensa, terminado, False, info
