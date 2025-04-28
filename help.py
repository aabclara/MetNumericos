def plotGraficos(f, intervalo, titulo=None, pontos=5000, figsize=(10,6)):
    import numpy as np
    import matplotlib.pyplot as plt
    """Gera o gráfico de uma função f(x) em um determinado intervalo.
    Parâmetros:
    - f: função a ser plotada
    - intervalo: tupla (x_min, x_max)
    - titulo: título do gráfico (opcional)
    - pontos: quantidade de pontos entre x_min e x_max (opcional)
    - figsize: tamanho da figura (opcional)
    """
    x_min, x_max = intervalo
    x = np.linspace(x_min, x_max, pontos)
    y = f(x)

    plt.figure(figsize=figsize)
    if titulo:
        plt.title(titulo)
    else:
        plt.title(f"Gráfico de f(x) no intervalo [{x_min}, {x_max}]")
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.grid(True)
    plt.axvline(x=0, color='black', linestyle='-')  # Eixo Y
    plt.axhline(y=0, color='black', linestyle='-')  # Eixo X
    plt.show()

def Bissecao(f, inicio, final, tolerancia=0.0009):
    contador = 0

    if f(inicio) * f(final) >= 0:
        return None, 0  # não há raiz garantida no intervalo

    while True:
        pontoMeio = (inicio + final) / 2
        resultado = f(pontoMeio)
        contador += 1

        if abs(resultado) < tolerancia:
            return pontoMeio, contador

        if resultado > 0:
            final = pontoMeio
        else:
            inicio = pontoMeio

def RegraDescartes(coeficientes):
    # Variação de sinais para raízes positivas
    variacoes_pos = 0
    for i in range(len(coeficientes) - 1):
        if coeficientes[i] * coeficientes[i + 1] < 0:
            variacoes_pos += 1

    # Variação de sinais para raízes negativas (substituindo x por -x)
    termos_neg = []
    grau = len(coeficientes) - 1
    for i in range(len(coeficientes)):
        coef = coeficientes[i]
        # Inverte o sinal para termos de grau ímpar (substitui x por -x)
        if grau % 2 != 0:
            coef *= -1
        termos_neg.append(coef)
        grau -= 1

    variacoes_neg = 0
    for i in range(len(termos_neg) - 1):
        if termos_neg[i] * termos_neg[i + 1] < 0:
            variacoes_neg += 1

    # Retorna as possíveis raízes positivas e negativas
    return (f"Possíveis raízes positivas: {variacoes_pos}\n"
            f"Possíveis raízes negativas: {variacoes_neg}")

def Secante(f,x0, x1, tol = 1e-6, max_iter = 100):
    for i in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0 = x1
        x1 = x2
    return None # se não converge

def falsaPosicao(f, a, b, tol = 1e-6, max_iter = 100):
    """
    Encontra a raiz, no intervalo [x0, x1], da equação definida em f.
    Parada: no máximo, max_iter, iterações ou diferença entre os limites do intervalo menor que tol.
    """
    if f(a) * f(b) >= 0:
        print("Erro! f(a) e f(b) devem ter sinais opostos!")
        return None
    
    for i in range(max_iter):
        c = (a * f(b) - b*f(a)) / (f(b)-f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        return None #não converge