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

# ==================================

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

# ==================================

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


# ==================================
# HELP DA PROVA ANTERIOR
# ==================================
"""
FUNÇÕES DISPONÍVEIS:

1. bissecao(f, a, b, tol=1e-6, max_iter=100)
    - Encontra uma raiz de f(x) no intervalo [a, b] usando o Método da Bisseção.
    - Parâmetros:
        f : função (ex: lambda x: x**2 - 4)
        a, b : extremos do intervalo com f(a) * f(b) < 0
        tol : tolerância para o erro
        max_iter : número máximo de iterações
    - Retorno: raiz aproximada de f(x) = 0

2. falsa_posicao(f, a, b, tol=1e-6, max_iter=100)
    - Método da Falsa Posição (Regula Falsi) para encontrar raiz de f(x).
    - Parâmetros iguais aos do método da bisseção.
    - Retorno: raiz aproximada de f(x) = 0

3. secante(f, x0, x1, tol=1e-6, max_iter=100)
    - Método da Secante para encontrar uma raiz de f(x), sem exigir sinal oposto.
    - Parâmetros:
        f : função (ex: lambda x: x**2 - 4)
        x0, x1 : aproximações iniciais
        tol, max_iter : precisão e limite de iterações
    - Retorno: raiz aproximada

4. regressao_linear(x, y)
    - Ajusta uma reta (y = b0 + b1 * x) aos dados usando fórmulas clássicas.
    - Parâmetros:
        x : lista ou array com os valores independentes
        y : lista ou array com os valores dependentes
    - Retorno:
        - função lambda para predição (reta ajustada)
        - R² (coeficiente de determinação)

5. regressao_linear_multipla(X, y)
    - Realiza regressão linear múltipla com várias variáveis independentes.
    - Parâmetros:
        X : matriz 2D com variáveis independentes (formato [n_amostras, n_variáveis])
        y : vetor com variável dependente (n_amostras,)
    - Retorno:
        - coeficientes do modelo (lista com os betas)
        - intercepto (β₀)
        - previsões ajustadas pelo modelo (array de y estimado)
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def bissecao(f, a, b, tol=1e-6, max_iter=1000):
    """
    f : function
        Função contínua para a qual se deseja encontrar a raiz.
    a : float
        Extremidade inferior do intervalo inicial.
    b : float
        Extremidade superior do intervalo inicial.
    tol : float, opcional
        Tolerância para o critério de parada (diferença máxima aceitável entre os extremos).
        Valor padrão é 1e-6.
    max_iter : int, opcional
        Número máximo de iterações permitidas. Valor padrão é 1000.
    ---------------
    def f(x):
        return x**3 - 4*x + 1
    raiz = bissecao(f, 0, 2)
    print(f"Raiz aproximada: {raiz}")
    """
    if f(a) * f(b) >= 0:
        print("O método da bisseção não garante raiz nesse intervalo.")
        return None

    iteracoes = 0
    while (b - a) / 2 > tol and iteracoes < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            print(f"Raiz de f(x) no intervalo é {c:.3f}")
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iteracoes += 1

    print(f"Raiz de f(x) no intervalo é {((a + b) / 2):.3f}")
    return (a + b) / 2


def grafico_bissecao(f, a, b):
    raiz = bissecao(f, a, b)
    if raiz is None:
        print("Não foi possível encontrar raiz.")
        return

    x = np.linspace(a-1, b+1, 500)
    y = [f(i) for i in x]

    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(raiz, f(raiz), 'ro', label=f'Raiz aprox: {raiz:.4f}')
    plt.title("Método da Bisseção")
    plt.legend()
    plt.grid()
    plt.show()

######################
def secante(f, x0, x1, tol=1e-6, max_iter=1000):
    """
    f : function
        Função contínua para a qual se deseja encontrar a raiz.
    x0 : float
        Primeiro valor inicial de aproximação.
    x1 : float
        Segundo valor inicial de aproximação.
    tol : float, opcional
        Tolerância para o critério de parada. Valor padrão é 1e-6.
    max_iter : int, opcional
        Número máximo de iterações permitidas. Valor padrão é 1000.
    """
    iteracoes = 0
    while abs(x1 - x0) > tol and iteracoes < max_iter:
        if f(x1) - f(x0) == 0:
            print("Divisão por zero na iteração, método da secante falhou.")
            return None
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x2
        iteracoes += 1
    print(f"Raiz de f(x) pelo método da secante: {x1:.3f} (em {iteracoes} iterações)")
    return x1, iteracoes

def grafico_secante(f, x0, x1):
    resultado = secante(f, x0, x1)
    if resultado is None:
        print("Método falhou.")
        return
    raiz, _ = resultado

    x = np.linspace(x0-1, x1+1, 500)
    y = [f(i) for i in x]

    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(raiz, f(raiz), 'ro', label=f'Raiz aprox: {raiz:.4f}')
    plt.title("Método da Secante")
    plt.legend()
    plt.grid()
    plt.show()


#############################
def falsa_posicao(f, a, b, tol=1e-6, max_iter=1000):
    """
    f : function
        Função contínua para a qual se deseja encontrar a raiz.
    a : float
        Extremidade inferior do intervalo inicial.
    b : float
        Extremidade superior do intervalo inicial.
    tol : float, opcional
        Tolerância para o critério de parada. Valor padrão é 1e-6.
    max_iter : int, opcional
        Número máximo de iterações permitidas. Valor padrão é 1000.
    """
    if f(a) * f(b) >= 0:
        print("O método da falsa posição não garante raiz nesse intervalo.")
        return None

    iteracoes = 0
    while abs(b - a) > tol and iteracoes < max_iter:
        fa = f(a)
        fb = f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        if abs(fc) < tol:
            print(f"Raiz de f(x) pelo método da falsa posição: {c:.3f} (em {iteracoes} iterações)")
            return c, iteracoes

        if fa * fc < 0:
            b = c
        else:
            a = c

        iteracoes += 1
    # return (a * f(b) - b * f(a)) / (f(b) - f(a)), iteracoes
    c = (a * f(b) - b * f(a)) / (f(b) - f(a))
    print(f"Raiz de f(x) pelo método da falsa posição: {c:.3f} (em {iteracoes} iterações)")
    return c, iteracoes

    

def grafico_falsa_posicao(f, a, b):
    resultado = falsa_posicao(f, a, b)
    if resultado is None:
        print("Não foi possível encontrar raiz.")
        return
    raiz, _ = resultado

    x = np.linspace(a-1, b+1, 500)
    y = [f(i) for i in x]

    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(raiz, f(raiz), 'ro', label=f'Raiz aprox: {raiz:.4f}')
    plt.title("Método da Falsa Posição")
    plt.legend()
    plt.grid()
    plt.show()

###################################
def regressao_linear(x, y):
    """
    x : list ou array-like de floats
        Valores independentes.
    y : list ou array-like de floats
        Valores dependentes.
    """
    n = len(x)
    if n != len(y) or n == 0:
        raise ValueError("Vetores x e y devem ter o mesmo tamanho e não estar vazios.")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerador = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominador = sum((x[i] - mean_x) ** 2 for i in range(n))

    if denominador == 0:
        raise ValueError("Variância de x é zero, não é possível ajustar uma reta.")

    b1 = numerador / denominador
    b0 = mean_y - b1 * mean_x

    # Função de predição
    reta = lambda x_val: b0 + b1 * x_val

    # Calcular R²
    ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((y[i] - reta(x[i])) ** 2 for i in range(n))

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"Equação ajustada: y = {b0:.4f} + {b1:.4f}x")
    print(f"Coeficiente de determinação R² = {r2:.2f}")
    return reta, r2

def grafico_regressao_linear(x, y):
    reta, r2 = regressao_linear(x, y)
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = [reta(i) for i in x_vals]

    plt.scatter(x, y, color='blue', label='Pontos')
    plt.plot(x_vals, y_vals, 'r-', label=f'Reta ajustada (R²={r2:.4f})')
    plt.title("Regressão Linear")
    plt.legend()
    plt.grid()
    plt.show()

#######################################
def regressao_polinomial(x, y, grau):
    """
    x : list ou array-like de floats
        Valores independentes.
    y : list ou array-like de floats
        Valores dependentes.
    grau : int
        Grau do polinômio a ser ajustado.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Vetores x e y devem ter o mesmo tamanho e não estar vazios.")

    # Construir matriz de Vandermonde para polinômios
    X = np.vander(x, grau + 1, increasing=True)  # colunas: x^0, x^1, ..., x^grau

    # Resolver sistema pelo método dos mínimos quadrados: beta = (X'X)^-1 X'y
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Função polinomial ajustada
    def poly_func(x_val):
        # calcula y = sum(coef[i] * x_val^i)
        x_val = np.array(x_val)
        # se x_val for escalar, funciona igual; se array, também funciona
        powers = np.vstack([x_val**i for i in range(grau + 1)])
        return np.dot(coef, powers)

    # Calcular R²
    y_pred = poly_func(x)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print("Equação ajustada (coeficientes do polinômio):")
    for i, c in enumerate(coef):
        print(f"β{i} = {c:.4f}")
    print(f"Coeficiente de determinação R² = {r2:.4f}")
    return poly_func, r2


def grafico_regressao_polinomial(x, y, grau):
    poly_func, r2 = regressao_polinomial(x, y, grau)
    x_vals = np.linspace(min(x), max(x), 300)
    y_vals = poly_func(x_vals)

    plt.scatter(x, y, color='blue', label='Pontos')
    plt.plot(x_vals, y_vals, 'r-', label=f'Polinômio grau {grau} (R²={r2:.4f})')
    plt.title(f"Regressão Polinomial (grau {grau})")
    plt.legend()
    plt.grid()
    plt.show()

###############################
def regressao_linear_multipla(X, y, retornar=True):
    """
    - X: array 2D (matriz) com variáveis independentes.
    - y: array 1D com variável dependente.
    ----
    - coeficientes: array com os coeficientes das variáveis independentes.
    - intercepto: valor do intercepto (β0)
    - previsoes: valores previstos pelo modelo.
    """
    # Cria o modelo
    modelo = LinearRegression()

    # Treina o modelo com os dados
    modelo.fit(X, y)

    coeficientes = modelo.coef_
    intercepto = modelo.intercept_
    previsoes = modelo.predict(X)

    termos = []
    for i, coef in enumerate(coeficientes):
        var = f"x{i+1}"
        sinal = "+" if coef >= 0 else "-"
        termos.append(f"{sinal} {abs(coef):.4f}*{var}")

    equacao = f"y = {intercepto:.4f} " + " ".join(termos)

    print("Modelo ajustado (Regressão Linear Múltipla):")
    print(equacao)

    print("---- Outros Dados -----")
    for i, coef in enumerate(coeficientes):
        print(f"Coeficiente β{i+1}: {coef:.4f}")
    print(f"Intercepto β0: {intercepto:.4f}")

    if retornar == False:
        return modelo, coeficientes, intercepto, previsoes, equacao

def grafico_regressao_linear_multipla(X, y):
    """
    Função para plotar a Regressão Linear Múltipla em gráfico 3D.
    
    Parâmetros:
    - X: array 2D com duas variáveis independentes.
    - y: array 1D com variável dependente.
    
    Exibe o gráfico 3D com os pontos e o plano ajustado.
    """
    # Cria e treina o modelo
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Obtém os coeficientes e intercepto
    b1, b2 = modelo.coef_
    b0 = modelo.intercept_

    # Cria malha de valores para o plano
    x_surf, y_surf = np.meshgrid(
        np.linspace(X[:,0].min(), X[:,0].max(), 10),
        np.linspace(X[:,1].min(), X[:,1].max(), 10)
    )
    
    # Calcula os valores previstos para a malha (plano)
    z_surf = b0 + b1 * x_surf + b2 * y_surf

    # Cria o gráfico 3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Plota os pontos
    ax.scatter(X[:,0], X[:,1], y, color='red', label='Dados')

    # Plota o plano
    ax.plot_surface(x_surf, y_surf, z_surf, color='lightblue', alpha=0.5)

    # Labels
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Regressão Linear Múltipla (Plano 3D)')
    ax.legend()

    plt.show()

##############################
def grafico_funcao_simples(f, x_min=-10, x_max=10, titulo="Gráfico de f(x)", num_pontos=500):
    """
    Plota o gráfico de uma função f(x) simples no intervalo [x_min, x_max].

    Parâmetros:
    - f : função (ex: lambda x: x**2 - 4)
    - x_min : limite inferior do eixo x (padrão: -10)
    - x_max : limite superior do eixo x (padrão: 10)
    - titulo : título do gráfico (padrão: "Gráfico de f(x)")
    - num_pontos : número de pontos a serem usados no gráfico (padrão: 500)

    Exemplo:
    >>> f = lambda x: x**3 - 2*x - 5
    >>> grafico_funcao_simples(f, x_min=0, x_max=3)
    """
    x = np.linspace(x_min, x_max, num_pontos)
    y = [f(i) for i in x]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True)
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()