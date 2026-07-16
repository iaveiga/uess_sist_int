import random
import itertools
import numpy as np

def generar_inventario_y_almacen(p_pasillos_hor, q_pasillos_ver, r_niveles, z_objetos_nivel, cap_almacen, num_objetos, l_max_stock):
    #Generar todas las Categorías y SKUs
    # Categorías: CATEG_PQ (P, Q del 0 al 9) -> 100 combinaciones
    categorias_pool = [f"CATEG_{p}{q}" for p in range(10) for q in range(10)]

    # SKUs: SK_M³N⁴ (M en A,B,C y N en 1-9) -> 3³ * 9⁴ = 27 * 6561 = 177,147 combinaciones
    m_letras = [''.join(x) for x in itertools.product('ABC', repeat=3)]
    n_numeros = [''.join(x) for x in itertools.product('123456789', repeat=4)]
    skus_pool = [f"SK_{m}{n}" for m in m_letras for n in n_numeros]

    # Tomar 'k' productos, máximo 2 objetos por producto
    promedio_stock = (1 + l_max_stock) / 2
    num_productos_necesarios = int(np.ceil(num_objetos / promedio_stock)) + 100
    skus_productos = random.sample(skus_pool, num_productos_necesarios)

    # Asignar stock aleatorio entre 1 y L a cada producto
    catalogo_maestro = {
        sku: {
            "categoria": random.choice(categorias_pool),
            "stock_disponible": random.randint(1, l_max_stock)  # Ajustado a L
        }
        for sku in skus_productos
    }

    # Generar el universo de objetos a colocar en el almacen
    lista_objetos_fisicos = []
    for sku, info in catalogo_maestro.items():
        for copia in range(info["stock_disponible"]):
            if len(lista_objetos_fisicos) < num_objetos:
                lista_objetos_fisicos.append(sku)

    #Barajar los objetos para colocarlos por orden de "llegada" y no por SKU
    random.shuffle(lista_objetos_fisicos)

    #Preparar el inventario del almacén

    # Estructura del inventario físico: {(coordenada_x, coordenada_y): {nivel: [lista_de_objetos]}}
    inventario_almacen = {}

    # Definimos dimensiones visuales de la cuadrícula (Pasillo - Percha - Pasillo)
    ancho_grid = (q_pasillos_ver * 2) + 1
    alto_grid = (p_pasillos_hor * 2) + 1
    mapa_navegacion = np.zeros((alto_grid, ancho_grid), dtype=int)

    for y in range(1, alto_grid, 2):
        for x in range(1, ancho_grid, 2):
            mapa_navegacion[y, x] = 1 # 1 representa estructura de percha (obstáculo)
            inventario_almacen[(x, y)] = {}

            for nivel in range(r_niveles):
                inventario_almacen[(x, y)][nivel] = []
                for slot in range(z_objetos_nivel):
                    # Si aún quedan objetos físicos por colocar, se asignan a la percha
                    if lista_objetos_fisicos:
                        sku_colocado = lista_objetos_fisicos.pop(0)
                    else:
                        sku_colocado = None # Slot vacío (Mínimo el 60% del almacén quedará así)
                    inventario_almacen[(x, y)][nivel].append(sku_colocado)

    return mapa_navegacion, inventario_almacen

def graficar_almacen_emoji(mapa_navegacion, pos_robot=None, pos_despacho=(0, 0)):
    #Toma un mapa de navegación (matriz numpy 1-0) y la grafica con emojis
    alto, ancho = mapa_navegacion.shape
    representacion = []
    #🤖: Robot despachador
    #🚛: Bahía de despacho
    #🗄️: Percha
    #⬜: Camino libre
    for y in range(alto):
        fila_emojis = []
        for x in range(ancho):
            if pos_robot and (x, y) == pos_robot:
                fila_emojis.append("🤖")
            elif (x, y) == pos_despacho:
                fila_emojis.append("🚛")
            elif mapa_navegacion[y, x] == 1:
                fila_emojis.append("🗄️")
            else:
                fila_emojis.append("⬜")

        representacion.append(" ".join(fila_emojis))
    print("\n".join(representacion))
