
import numpy as np



####################
def Bissecao(f, inicio, final, tolerancia=1e-5, max_iteracoes=1000):
    """
    Encontra uma raiz de f(x) no intervalo [inicio, final] usando Bisseção.

    Retorna:
        - raiz: O valor aproximado da raiz.
        - iteracoes: Quantidade de iterações.
        - None, 0: Se não houver raiz garantida no intervalo inicial.
    """
    contador = 0

    if f(inicio) * f(final) >= 0:
        print("Erro: f(inicio) e f(final) devem ter sinais opostos.")
        iteracoes = 0
    while iteracoes < max_iteracoes and abs(final - inicio) > tolerancia:
        pontoMeio = (inicio + final) / 2
        resultado = f(pontoMeio)
        iteracoes += 1

        if abs(resultado) < tolerancia:
            return pontoMeio, contador

        if resultado > 0:
            final = pontoMeio
        else:
            inicio = pontoMeio
        return (inicio + final) / 2, iteracoes

######################
def falsa_posicao(f, a, b, tol=1e-6, max_iter=100):
    """
    Encontra uma raiz de f(x) no intervalo [a, b] usando Falsa Posição.

    Retorna:
        - raiz: O valor aproximado da raiz.
        - iteracoes: Quantidade de iterações.
        - None: Se f(a) e f(b) não tiverem sinais opostos.
    """
    qtd_iteracao = 0

    if f(a) * f(b) >= 0:
        print("Erro: f(a) e f(b) devem ter sinais opostos.")
        return None, 0

    for i in range(max_iter):
        # Fórmula da Falsa Posição
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        qtd_iteracao += 1

        if abs(f(c)) < tol:
            return c, qtd_iteracao

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    print(f"Aviso: O método da Falsa Posição não convergiu após {max_iter} iterações.")
    return None, qtd_iteracao # Retorna None e a contagem de iterações se não convergir

####################
def secante(f, x0, x1, tol = 1e-6, max_iter = 100):
    """
    Encontra uma raiz de f(x) usando o Método da Secante.

    Parâmetros:
        - f: A função Python.
        - x0, x1: Dois pontos iniciais próximos à raiz.
        - tol (opcional): Tolerância para a convergência.
        - max_iter (opcional): Número máximo de iterações.

    Retorna:
        - raiz: O valor aproximado da raiz.
        - None: Se não convergir dentro do número máximo de iterações.
    """
    for i in range(max_iter):
        # Evita divisão por zero se f(x1) for igual a f(x0)
        if abs(f(x1) - f(x0)) < 1e-10: # Usar uma pequena tolerância para comparação de floats
             print(f"Aviso: Denominador próximo de zero na iteração {i}. Interrompendo.")
             return None

        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0 = x1
        x1 = x2
    print(f"Aviso: O método da Secante não convergiu após {max_iter} iterações.")
    return None # se não converge

#######################

def ajuste_linear(x, y):
    """
    Calcula os coeficientes de um ajuste linear (reta) y = b0 + b1*x
    para os dados (x, y) usando o método dos mínimos quadrados.

    Parâmetros:
        - x: Array de valores da variável independente.
        - y: Array de valores da variável dependente.

    Retorna:
        - (b0, b1): Tupla contendo os coeficientes do ajuste linear
                      (intercepto b0 e inclinação b1).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(x)

    matriz = np.array([
        [n, np.sum(x)],
        [np.sum(x), np.sum(x**2)]
    ])

    vetor = np.array([
        np.sum(y),
        np.sum(x * y)
    ])
    # Alternativamente, para um vetor coluna 2x1:
    # vetor = np.array([[np.sum(y)],
    #                   [np.sum(x*y)]])

    b0, b1 = np.linalg.solve(matriz, vetor)
    return (b0, b1)

########################

def ajuste_quadratico(x, y):
    """
    Calcula os coeficientes de um ajuste quadrático (parábola)
    y = b0 + b1*x + b2*x^2 para os dados (x, y) usando o método
    dos mínimos quadrados.

    Parâmetros:
        - x: Array de valores da variável independente.
        - y: Array de valores da variável dependente.

    Retorna:
        - (b0, b1, b2): Tupla contendo os coeficientes do ajuste quadrático
                          (b0, b1, b2).
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Corrected matriz definition
    matriz = np.array([
        [len(x), np.sum(x), np.sum(x**2)],
        [np.sum(x), np.sum(x**2), np.sum(x**3)],
        [np.sum(x**2), np.sum(x**3), np.sum(x**4)]
    ])

    # Corrected vetor definition
    vetor = np.array([
        np.sum(y),
        np.sum(x * y),
        np.sum(x**2 * y)
    ]) # .T is optional here, np.linalg.solve handles 1D array for the second arg

    b0, b1, b2 = np.linalg.solve(matriz, vetor)
    return (b0, b1, b2)