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

def sistemas_triangularesInferior(L, C):
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

def sistema_triangular_superior(U, C):
    n = U.shape[0]  
    x = np.zeros(n) 

    for i in range(n - 2, -1, -1):
        soma = 0

        for j in range(i + 1, n):
            soma += U[i, j] * x[j]

        if U[i, i] == 0:
            raise ValueError(f"Elemento U[{i},{i}] na diagonal é zero, divisão por zero impossível.")
        x[i] = (C[i] - soma) / U[i, i]

    return x

# solucao2 = sistema_triangular_superior(U_exemplo2, C_exemplo2)
# print("A solução do sistema linear triangular superior (x) é:")
# print(solucao2)

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

#################################################################################

def decomposicao_lu(matriz_entrada):
    numero_linhas = matriz_entrada.shape[0]  
    
    matriz_inferior_L = np.eye(numero_linhas) 
    matriz_superior_U = matriz_entrada.copy().astype(float)  

    for coluna_pivo in range(numero_linhas - 1):  
        elemento_pivo = matriz_superior_U[coluna_pivo, coluna_pivo]

        if elemento_pivo == 0:
            print(f"Erro: Elemento pivô zero na posição ({coluna_pivo}, {coluna_pivo}). "
                  "A decomposição LU pode não ser possível sem pivoteamento.")
            return None, None

        for linha_atual in range(coluna_pivo + 1, numero_linhas):  
            fator_multiplicador = matriz_superior_U[linha_atual, coluna_pivo] / elemento_pivo
            
            matriz_inferior_L[linha_atual, coluna_pivo] = fator_multiplicador
            
            matriz_superior_U[linha_atual, coluna_pivo:] = matriz_superior_U[linha_atual, coluna_pivo:] - fator_multiplicador * matriz_superior_U[coluna_pivo, coluna_pivo:]
            
    return matriz_inferior_L, matriz_superior_U

#################################################################################

def calcular_determinante(matriz_entrada):
    num_linhas, num_colunas = matriz_entrada.shape
    if num_linhas != num_colunas:
        print("Erro: O determinante só pode ser calculado para matrizes quadradas.")
        return None
    
    matriz_temp = matriz_entrada.copy().astype(float)
    
    determinante = 1.0 

    for k in range(num_linhas): 
        elemento_pivo = matriz_temp[k, k]

        if abs(elemento_pivo) < 1e-9: 
            print(f"Aviso: Elemento pivô muito pequeno ou zero na posição ({k}, {k}). "
                  "A matriz pode ser singular (determinante ~ 0).")
            return 0.0 

        determinante *= elemento_pivo 
        for i in range(k + 1, num_linhas):
            fator_multiplicador = matriz_temp[i, k] / elemento_pivo
            
            matriz_temp[i, k:] = matriz_temp[i, k:] - fator_multiplicador * matriz_temp[k, k:]
            
    return determinante

#################################################################################
# Interpolação
# - Newton: Diferenças Divididas
#       - Identificar como preencher a tabela das diferenças divididas usando a função.
# - Polinomio de Lagrange
# Qual o polinômio
# Valor do polinômio para um dado X.

# Solução de Sistemas de Equações 
# Eliminação de Gauss: Transformar a matriz em um sistema triangular superior ou inferior
#   - Matrizes e operações elementares
#   - Substituição sucessiva ou retrosubstituição
# Decomposição LU
#   - Qualquer matriz definida positiva
#       - Elemento da diagonal é 0
# Decomposição de Cholesky
#   - Qualquer matriz definida positiva e simêtrica
#
# Verificar se a matriz é definida positiva por cholesky ou LU: Algum elemento da diagonal na decomposição é 0

