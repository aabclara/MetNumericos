import numpy as np

def polinomio_lagrange(coordenadas_x, coordenadas_y, x):
    array_coordenadas_x = np.asarray(coordenadas_x, dtype=float)
    array_coordenadas_y = np.asarray(coordenadas_y, dtype=float)

    num_pontos = len(array_coordenadas_x)

    if num_pontos != len(array_coordenadas_y):
        raise ValueError("O número de coordenadas x e y deve ser o mesmo!")

    if len(np.unique(array_coordenadas_x)) != num_pontos:
        raise ValueError("Os valores de x das coordenadas devem ser distintos para a interpolação de Lagrange!")

    valor_polinomio_lagrange = 0.0  

    for i in range(num_pontos):
        termo_produtorio = 1.0

        for j in range(num_pontos):
            if i != j:  
                termo_atual = (x - array_coordenadas_x[j]) / \
                              (array_coordenadas_x[i] - array_coordenadas_x[j])
                termo_produtorio *= termo_atual
        
        valor_polinomio_lagrange += array_coordenadas_y[i] * termo_produtorio

    return valor_polinomio_lagrange

##################################################################################

def divisao_diferencas(x, y):
    n = len(x)
    coeficientes = list(y)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coeficientes[i] = (coeficientes[i] - coeficientes[i-1]/(x[i]-x[i-j]))
    return coeficientes