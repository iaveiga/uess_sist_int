import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

class AlmacenEnv(gym.Env):
    def __init__(self, mapa, inventario, catalogo_skus, sku_objetivo, max_pasos=400):
        super(AlmacenEnv, self).__init__()
        
        self.mapa = mapa
        self.inventario_original = copy.deepcopy(inventario) 
        self.inventario = copy.deepcopy(inventario)
        
        self.alto, self.ancho = mapa.shape
        self.catalogo_skus = list(catalogo_skus)
        self.num_skus = len(self.catalogo_skus)
        self.sku_objetivo = sku_objetivo
        self.id_objetivo = self.catalogo_skus.index(sku_objetivo)
        
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
        
        # Coordenadas objetivo que se calcularán dinámicamente
        self.obj_x = 0
        self.obj_y = 0
        
        self.reset()

    def _actualizar_coordenada_objetivo(self):
        """Busca en el inventario la coordenada real (x, y) del SKU objetivo."""
        for coord, niveles in self.inventario.items():
            for nivel, skus in niveles.items():
                if self.sku_objetivo in skus:
                    self.obj_x, self.obj_y = coord
                    return
        # Si por alguna razón no se encuentra, por defecto apunta al origen
        self.obj_x, self.obj_y = 0, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_x = 0
        self.robot_y = 0
        self.theta = 0  # 0: Norte, 1: Este, 2: Sur, 3: Oeste
        self.tiene_producto = 0 
        self.pasos_actuales = 0
        
        self.inventario = copy.deepcopy(self.inventario_original)
        self._actualizar_coordenada_objetivo()
        
        estado_inicial = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        return estado_inicial, {}

    def step(self, action):
        self.pasos_actuales += 1
        prev_x, prev_y = self.robot_x, self.robot_y
        
        # Penalización base por paso (incentivo de tiempo)
        recompensa = -1
        terminado = False
        truncado = False
        
        dist_previa = abs(prev_x - self.obj_x) + abs(prev_y - self.obj_y)

        if action == 0:  # Avanzar
            if self.theta == 0:    # Norte
                self.robot_y = max(0, self.robot_y - 1)
            elif self.theta == 1:  # Este
                self.robot_x = min(self.ancho - 1, self.robot_x + 1)
            elif self.theta == 2:  # Sur
                self.robot_y = min(self.alto - 1, self.robot_y + 1)
            elif self.theta == 3:  # Oeste
                self.robot_x = max(0, self.robot_x - 1)
                
            # Verificar si chocó contra una percha (valor 1) o límites
            if self.mapa[self.robot_y, self.robot_x] == 1:
                recompensa = -15  
                self.robot_x, self.robot_y = prev_x, prev_y  
            elif self.robot_x == prev_x and self.robot_y == prev_y:
                recompensa = -10  
                
        elif action == 1:  # Girar Izquierda
            self.theta = (self.theta - 1) % 4
            recompensa = -1.5   # Penalización intermedia para evitar giros infinitos
            
        elif action == 2:  # Girar Derecha
            self.theta = (self.theta + 1) % 4
            recompensa = -1.5   
            
        elif action == 3:  # Extraer
            encontrado = False
            dx, dy = 0, 0
            if self.theta == 0: dy = -1
            elif self.theta == 1: dx = 1
            elif self.theta == 2: dy = 1
            elif self.theta == 3: dx = -1
            
            px, py = self.robot_x + dx, self.robot_y + dy
            
            # Verifica los límites y si tiene una percha en frente
            if 0 <= px < self.ancho and 0 <= py < self.alto and self.mapa[py, px] == 1:
                percha = self.inventario.get((px, py), {}) # Estructura: {nivel_0: [skus], nivel_1: [skus]...}
                
                # Recorre los R niveles (0 a 3) #harcoded por ahora
                for nivel in range(4):  # R = 4 niveles
                    lista_skus = percha.get(nivel, [])
                    
                    # Busca el SKU objetivo en los Z objetos de este nivel
                    if self.sku_objetivo in lista_skus:
                        idx = lista_skus.index(self.sku_objetivo)
                        
                        # Retirar una unidad del stock decrementando o dejándolo en None
                        lista_skus[idx] = None  
                        self.tiene_producto = 1
                        recompensa = 250  # Encontró el objeto
                        encontrado = True
                        break # Salimos del bucle de niveles
                
            if not encontrado:
                recompensa = -30  # Castigo por fallar la extracción

 
        if self.tiene_producto == 0:
            # Recompensa si se está acercando al SKU objetivo
            dist_nueva = abs(self.robot_x - self.obj_x) + abs(self.robot_y - self.obj_y)
            if dist_nueva < dist_previa:
                recompensa += 2.0  # Incentivo positivo por avanzar en la dirección correcta
            elif dist_nueva > dist_previa:
                recompensa -= 1.0  # Penalización por alejarse del objetivo
        else:
            # Si ya tiene el producto, premiamos que se acerque al punto de retorno (0,0)
            dist_retorno_previa = abs(prev_x - 0) + abs(prev_y - 0)
            dist_retorno_nueva = abs(self.robot_x - 0) + abs(self.robot_y - 0)
            if dist_retorno_nueva < dist_retorno_previa:
                recompensa += 2.0  # Incentivo por volver a base
            elif dist_retorno_nueva > dist_retorno_previa:
                recompensa -= 1.0

        
        if self.robot_x == 0 and self.robot_y == 0:
            if self.tiene_producto == 1:
                recompensa = 500  # Premio gigante por completar el circuito completo
                terminado = True
            else:
                recompensa = -50  # Castigo fuerte por volver a base con las manos vacías

        # --- 5. LÍMITE DE PASOS ---
        if self.pasos_actuales >= self.max_pasos:
            truncado = True

        info = {}
        estado = (self.robot_x, self.robot_y, self.theta, self.tiene_producto, self.id_objetivo)
        
        return estado, recompensa, terminado, truncado, info
