import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plotGraficos(f, intervalo, titulo=None, pontos=5000, figsize=(9,5)):
    """
    Gera o gráfico de uma função f(x) em um intervalo.

    Parâmetros:
        - f: A função Python a ser plotada (ex: lambda x: x**2 - 4).
        - intervalo: Tupla (x_min, x_max) definindo o eixo X.
        - titulo (opcional): Título para o gráfico.
    """
    x_min, x_max = intervalo
    x = np.linspace(x_min, x_max, pontos)
    y = f(x)

    plt.figure(figsize=figsize)
    plt.title(titulo if titulo else f"Gráfico no intervalo [{x_min}, {x_max}]")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.grid(True)
    plt.axvline(x=0, color='black', linestyle='-')  # Eixo Y
    plt.axhline(y=0, color='black', linestyle='-')  # Eixo X
    plt.show()

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

def RegraDescartes(coeficientes):
    """
    Aplica a Regra de Sinais de Descartes para estimar raízes positivas/negativas.

    Parâmetros:
        - coeficientes: Lista ou array dos coeficientes do polinômio,
                        do maior grau para o menor (ex: [1, -2, -3] para x^2 - 2x - 3).

    Retorna:
        - string: Texto com o número possível de raízes positivas e negativas.
    """
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

    return (f"Possíveis raízes positivas: {variacoes_pos}\n"
            f"Possíveis raízes negativas: {variacoes_neg}")


def Secante(f, x0, x1, tol = 1e-6, max_iter = 100):
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



def ajustar_reta(x, y):
    """
    Ajusta uma reta (modelo linear) aos dados (x, y).

    Retorna:
        - func_reta: Uma função para calcular 'y' dado um 'x' na reta.
        - r2: O R² do ajuste.
    """
    n = len(x)
    b1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
         (n * np.sum(x**2) - np.sum(x)**2)
    b0 = (np.sum(y) / n) - b1 * (np.sum(x) / n)

    func_reta = lambda val_x: b1 * val_x + b0
    y_pred = func_reta(x)
    r2 = r2_score(y, y_pred)
    return func_reta, r2

def ajustar_polinomio(x, y, grau):
    """
    Ajusta um polinômio de um certo grau aos dados (x, y).

    Parâmetros:
        - grau (int): 1 para linear, 2 para quadrático, 3 para cúbico.

    Retorna:
        - func_polinomio: Uma função para calcular 'y' dado um 'x' no polinômio.
        - r2: O R² do ajuste.
    """
    coefs = np.polyfit(x, y, grau)
    func_polinomio = np.poly1d(coefs)
    y_pred = func_polinomio(x)
    r2 = r2_score(y, y_pred)
    return func_polinomio, r2

def comparar_modelos_grafico(x_dados, y_dados, modelos):
    """
    Cria um gráfico comparando os dados e os modelos ajustados.

    Parâmetros:
        - modelos (list): Lista de dicionários, cada um com:
            - 'nome' (str): Nome do modelo (ex: 'Linear', 'Quadrático').
            - 'funcao' (function): A função do modelo (ajustada por ajustar_reta ou ajustar_polinomio).
            - 'r2' (float): O R² do modelo (para exibir na legenda).
            - 'cor' (str, opcional): Cor da linha (ex: 'blue').
            - 'estilo' (str, opcional): Estilo da linha (ex: '--').
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_dados, y_dados, color='red', label='Dados Originais', zorder=5)

    x_plot = np.linspace(min(x_dados) - 0.1 * np.ptp(x_dados),
                         max(x_dados) + 0.1 * np.ptp(x_dados), 200)

    for m in modelos:
        y_pred_plot = m['funcao'](x_plot)
        label = f"{m['nome']} (R²={m['r2']:.3f})"
        plt.plot(x_plot, y_pred_plot, color=m.get('cor', 'blue'),
                 linestyle=m.get('estilo', '-'), label=label)

    plt.title('Comparação dos Modelos de Regressão')
    plt.xlabel('Variável Independente (x)')
    plt.ylabel('Variável Dependente (y)')
    plt.legend()
    plt.grid(True)
    plt.show()

'''
# --- EXPLICANDO COMO USAR (Este bloco só roda se você executar este arquivo) ---
if __name__ == '__main__':
    print("--- Guia Rápido para a Avaliação ---")
    print("\nPara usar estas funções, salve este código como 'meu_help_simples.py' ")
    print("e importe no seu arquivo principal assim:")
    print("from meu_help_simples import ajustar_reta, ajustar_polinomio, comparar_modelos_grafico")

    # --- Exemplo com seus dados ---
    print("\n--- Exemplo de Uso com seus Dados de Peixes ---")
    x = np.array([13, 15, 16, 21, 22, 23, 25, 29, 30, 31, 36, 40, 42, 55, 60, 62, 64, 70, 72, 100, 130])
    y = np.array([11, 10, 11, 12, 12, 13, 12, 14, 16, 17, 13, 14, 22, 14, 21, 21, 24, 17, 23, 23, 34])

    # 1. Ajustar Modelos:
    print("\n1. Ajustando os Modelos:")
    func_linear, r2_linear = ajustar_reta(x, y)
    print(f"   Modelo Linear ajustado. R²: {r2_linear:.4f}")

    func_quad, r2_quad = ajustar_polinomio(x, y, 2)
    print(f"   Modelo Quadrático ajustado. R²: {r2_quad:.4f}")

    func_cubic, r2_cubic = ajustar_polinomio(x, y, 3)
    print(f"   Modelo Cúbico ajustado. R²: {r2_cubic:.4f}")

    # 2. Prever valores (Exemplo: para x = 50):
    print("\n2. Previsão para x = 50:")
    print(f"   Linear: y = {func_linear(50):.2f}")
    print(f"   Quadrático: y = {func_quad(50):.2f}")
    print(f"   Cúbico: y = {func_cubic(50):.2f}")

    # 3. Comparar R² para escolher o melhor modelo:
    print("\n3. Escolhendo o Melhor Modelo (pelo R²):")
    modelos_r2 = {
        'Linear': r2_linear,
        'Quadrático': r2_quad,
        'Cúbico': r2_cubic
    }
    melhor_modelo_nome = max(modelos_r2, key=modelos_r2.get)
    print(f"   O modelo com maior R² é o {melhor_modelo_nome} (R²: {modelos_r2[melhor_modelo_nome]:.4f}).")

    # 4. Gerar Gráfico:
    print("\n4. Gerando Gráfico Comparativo...")
    modelos_para_plot = [
        {'nome': 'Linear', 'funcao': func_linear, 'r2': r2_linear, 'cor': 'blue', 'estilo': '--'},
        {'nome': 'Quadrático', 'funcao': func_quad, 'r2': r2_quad, 'cor': 'green', 'estilo': '-'},
        {'nome': 'Cúbico', 'funcao': func_cubic, 'r2': r2_cubic, 'cor': 'purple', 'estilo': ':'}
    ]
    comparar_modelos_grafico(x, y, modelos_para_plot)
    print("   Gráfico gerado. Verifique a janela pop-up.")

    print("\n--- Fim do Guia Rápido ---")

'''