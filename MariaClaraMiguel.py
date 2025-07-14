import numpy as np

# ========================
# INTERPOLAÇÃO - NEWTON
# ========================

def diferencas_divididas(x, y):
    '''
    calcula a tabela de diferenças divididas de Newton a partir de pontos (x, y)

    parâmetros:
    x : lista ou array com os valores de x
    y : lista ou array com os valores correspondentes de f(x)

    retorna:
    um vetor com os coeficientes do polinômio de Newton
    '''
    n = len(x)
    tabela = np.zeros((n, n))
    tabela[:, 0] = y

    # preenche a tabela com as diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            tabela[i][j] = (tabela[i + 1][j - 1] - tabela[i][j - 1]) / (x[i + j] - x[i])

    print("Tabela de Diferenças Divididas:")
    print(tabela)

    return tabela[0]  # retorna os coeficientes do polinômio

def avalia_newton(x, x_ponto, coeficientes):
    '''
    avalia o polinômio de Newton, gerado com as diferenças divididas, em um ponto específico

    parâmetros:
    x : lista dos pontos originais utilizados na interpolação
    x_ponto : ponto no qual o polinômio será avaliado
    coeficientes : vetor dos coeficientes obtidos da função diferencas_divididas

    retorna:
    valor numérico do polinômio interpolador avaliado em x_ponto
    '''
    n = len(coeficientes)
    resultado = coeficientes[0]
    produto = 1.0
    for i in range(1, n):
        produto *= (x_ponto - x[i - 1])
        resultado += coeficientes[i] * produto
    print(f"O valor do polinômio de Newton em x = {x_ponto} é {resultado}")
    return resultado


# ========================
# INTERPOLAÇÃO - LAGRANGE
# ========================

def polinomio_lagrange(x, y, x_ponto):
    '''
    calcula o valor do polinômio de Lagrange que interpola os pontos (x, y) em um ponto específico

    parâmetros:
    x : lista ou array de abscissas
    y : lista ou array de ordenadas
    x_ponto : ponto no qual deseja-se calcular o valor do polinômio interpolador

    retorna:
    valor do polinômio de Lagrange no ponto x_ponto
    '''
    n = len(x)
    resultado = 0.0
    for i in range(n):
        termo = y[i]
        for j in range(n):
            if j != i:
                termo *= (x_ponto - x[j]) / (x[i] - x[j])
        resultado += termo

    print(f"O valor do polinômio de Lagrange em x = {x_ponto} é {resultado}")
    return resultado


# =======================================
# SOLUÇÃO DE SISTEMAS - ELIMINAÇÃO DE GAUSS
# =======================================

def eliminacao_gauss(matriz, termos):
    ''' 
    transforma o sistema linear Ax = b em uma forma triangular superior usando a eliminação de Gauss

    parâmetros:
    matriz : matriz dos coeficientes A (numpy array)
    termos : vetor b dos termos independentes

    retorna:
    a matriz A transformada em triangular superior e o vetor b modificado
    '''
    A = matriz.astype(float)
    b = termos.astype(float)
    n = len(b)

    # operações elementares para triangular superior
    for i in range(n):
        for j in range(i+1, n):
            fator = A[j][i] / A[i][i]
            A[j, i:] -= fator * A[i, i:]
            b[j] -= fator * b[i]

    print("Matriz após Eliminação de Gauss:")
    print(A)
    print("Vetor de termos independentes atualizado:")
    print(b)

    return A, b


def substituicao_reversa(A, b):
    '''
    transforma o sistema linear Ax = b em uma forma triangular superior usando a eliminação de Gauss

    parâmetros:
    matriz : matriz dos coeficientes A (numpy array)
    termos : vetor b dos termos independentes

    retorna:
    a matriz A transformada em triangular superior e o vetor b modificado
    '''

    n = len(b)
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]

    print("Solução do sistema via substituição reversa:")
    print(x)
    return x


# ============================
# DECOMPOSIÇÃO LU (Doolittle)
# ============================

def decomposicao_LU(A):
    '''
    realiza a decomposição de uma matriz A em duas matrizes: L (inferior) e U (superior), usando o método de Doolittle

    parâmetros:
    A : matriz dos coeficientes (numpy array)

    retorna:
    duas matrizes, L e U, tal que A = L @ U
    '''

    n = len(A)
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (A[j][i] - np.dot(L[j, :i], U[:i, i])) / U[i][i]

    print("Matriz L:")
    print(L)
    print("Matriz U:")
    print(U)

    return L, U


# ===============================
# DECOMPOSIÇÃO DE CHOLESKY
# ===============================

def cholesky(A):
    '''
    realiza a decomposição de Cholesky de uma matriz A, válida apenas para matrizes simétricas e definidas positivas

    parâmetros:
    A : matriz dos coeficientes (numpy array), simétrica e definida positiva

    retorna:
    matriz L tal que A = L @ L.T
    '''
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            soma = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - soma)
            else:
                L[i][j] = (A[i][j] - soma) / L[j][j]

    print("Matriz L da decomposição de Cholesky:")
    print(L)
    return L


# ====================================
# VERIFICAÇÃO DE MATRIZ DEFINIDA POSITIVA
# ====================================

def definida_positiva_por_cholesky(A):
    '''
    verifica se uma matriz é definida positiva tentando aplicar a decomposição de Cholesky

    parâmetros:
    A : matriz (numpy array)

    retorna:
    True se a matriz é definida positiva, False caso contrário
    '''

    try:
        L = cholesky(A)
        if np.any(np.diag(L) == 0):
            print("A matriz NÃO é definida positiva (diagonal da Cholesky tem zero).")
            return False
        print("A matriz é definida positiva (Cholesky válida).")
        return True
    except Exception as e:
        print(f"A matriz NÃO é definida positiva (falha na Cholesky): {e}")
        return False
    
# ==================================
# INTEGRAÇÃO NUMÉRICA - SIMPSON 1/3
# ==================================    

def integral_trapezio(funcao, a, b, n):
    '''
    aproxima a integral definida da função no intervalo [a, b] usando a regra do trapézio (composta)

    parâmetros:
    funcao : função matemática a ser integrada (deve ser uma função Python, ex: lambda x: x**2)
    a : limite inferior do intervalo de integração
    b : limite superior do intervalo de integração
    n : número de subintervalos (quanto maior, mais precisa a aproximação)

    retorna:
    valor aproximado da integral
    '''
    h = (b - a) / n
    soma = funcao(a) + funcao(b)

    for i in range(1, n):
        xi = a + i * h
        soma += 2 * funcao(xi)

    resultado = (h / 2) * soma
    print(f"Integral aproximada pelo método do trapézio em [{a}, {b}] com {n} subintervalos: {resultado}")
    return resultado

# ==================================
# INTEGRAÇÃO NUMÉRICA - SIMPSON 1/3
# ==================================

def integral_simpson(funcao, a, b, n):
    '''
    aproxima a integral definida da função no intervalo [a, b] usando a regra de Simpson 1/3 (composta)

    parâmetros:
    funcao : função matemática a ser integrada (deve ser uma função Python, ex: lambda x: x**2)
    a : limite inferior do intervalo de integração
    b : limite superior do intervalo de integração
    n : número de subintervalos (deve ser PAR)

    retorna:
    valor aproximado da integral
    '''
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para o método de Simpson.")

    h = (b - a) / n
    soma = funcao(a) + funcao(b)

    for i in range(1, n):
        xi = a + i * h
        peso = 4 if i % 2 != 0 else 2
        soma += peso * funcao(xi)

    resultado = (h / 3) * soma
    print(f"Integral aproximada pelo método de Simpson em [{a}, {b}] com {n} subintervalos: {resultado}")
    return resultado