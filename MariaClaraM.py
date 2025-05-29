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