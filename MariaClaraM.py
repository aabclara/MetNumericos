import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def bissecao(f, a, b, tol=1e-6, max_iter=1000):
    """
    Encontra uma raiz aproximada de uma função contínua f(x) no intervalo [a, b]
    usando o método da bisseção.
    -----------
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
    --------------
    O método verifica se há mudança de sinal no intervalo [a, b].
    Em seguida, divide repetidamente o intervalo ao meio e seleciona o subintervalo
    onde ocorre a troca de sinal, até que a largura do intervalo seja menor que a tolerância
    ou o número máximo de iterações seja atingido.
    --------
    float ou None
        Retorna a raiz aproximada encontrada ou None caso a condição inicial
        f(a) * f(b) >= 0 não seja satisfeita (o método da bisseção não garante raiz nesse caso).
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
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iteracoes += 1

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
    Encontra uma raiz aproximada de uma função contínua f(x) usando o método da secante.
    -----------
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
    --------
    tuple (float, int)
        - Raiz aproximada encontrada.
        - Número de iterações realizadas.
    """
    iteracoes = 0
    while abs(x1 - x0) > tol and iteracoes < max_iter:
        if f(x1) - f(x0) == 0:
            print("Divisão por zero na iteração, método da secante falhou.")
            return None
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x2
        iteracoes += 1

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
    Encontra uma raiz aproximada de uma função contínua f(x) no intervalo [a, b]
    usando o método da falsa posição (regula falsi).
    -----------
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
    --------
    tuple (float, int) ou None
        - Raiz aproximada encontrada.
        - Número de iterações realizadas.
        Retorna None se f(a) * f(b) >= 0 (quando o método não garante raiz no intervalo).
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
            return c, iteracoes

        if fa * fc < 0:
            b = c
        else:
            a = c

        iteracoes += 1

    return (a * f(b) - b * f(a)) / (f(b) - f(a)), iteracoes

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
    Ajusta uma reta (modelo de regressão linear simples) aos dados (x, y) usando
    as fórmulas clássicas para calcular os coeficientes b0 e b1.
    -----------
    x : list ou array-like de floats
        Valores independentes.
    y : list ou array-like de floats
        Valores dependentes.
    --------
    tuple:
        - função lambda que calcula y dado um x usando a reta ajustada.
        - coeficiente de determinação R² do ajuste (quanto mais prox de 1 melhor)
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
    Ajusta um polinômio de grau 'grau' aos dados (x, y) usando regressão polinomial.
    -----------
    x : list ou array-like de floats
        Valores independentes.
    y : list ou array-like de floats
        Valores dependentes.
    grau : int
        Grau do polinômio a ser ajustado.
    --------
    tuple:
        - função que calcula y dado um x usando o polinômio ajustado.
        - coeficiente de determinação R² do ajuste.
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
