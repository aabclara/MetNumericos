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

##################################################################################

def diferencas_divididas(pontos_x, valores_y):
    n = len(pontos_x)
    if n != len(valores_y):
        raise ValueError("Os arrays 'pontos_x' e 'valores_y' devem ter o mesmo tamanho.")

    diferencas = np.full((n, n), np.nan)
    
    diferencas[:, 0] = valores_y


    for j in range(1, n):  
        for i in range(n - j):  
            diferencas[i, j] = (diferencas[i + 1, j - 1] - diferencas[i, j - 1]) / \
                                      (pontos_x[i + j] - pontos_x[i])
    return diferencas

resultados = diferencas_divididas(x, y)
#
#print("Tabela de Diferenças Divididas Completa:")
#for linha in resultados:
#    linha_formatada = []
#    for elemento in linha:
#        if np.isnan(elemento):
#            linha_formatada.append("    X  ") 
#        else:
#            linha_formatada.append(f"{elemento:7.3f}")
#    print(" ".join(linha_formatada))

##################################################################################

def sistemas_triangulares(L, C):
    n = L.shape[0]   
    x = np.zeros(n) 

    if L[0, 0] == 0:
        raise ValueError("Elemento L[0,0] é zero, divisão por zero impossível.")
    x[0] = C[0] / L[0, 0]

    for i in range(1, n):
        soma = 0
        for j in range(i):
            soma += L[i, j] * x[j]
        if L[i, i] == 0:
            raise ValueError(f"Elemento L[{i},{i}] é zero, divisão por zero impossível.")
        x[i] = (C[i] - soma) / L[i, i]

    return x

##################################################################################

def integral(func, inferior, superior, n):
    if n <= 0:
        raise ValueError("O número de subintervalos (n) deve ser um inteiro positivo.")

    h = (superior - inferior) / n 
    soma_areas = func(inferior) + func(superior)

    for i in range(1, n):
        x_i = inferior + i * h
        soma_areas += 2 * func(x_i)

    integral_aproximada = (h / 2) * soma_areas

    return integral_aproximada