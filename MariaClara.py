# Importação das bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Função que plota o gráfico de uma função f qualquer
# f - função
# a, b - variavéis
def grafico(f, a, b, n, title = " "):
    x = np.linspace(a,b,n)
    y = f(x)
    plt.figure(figsize=(10,6))
    plt.title(title) # plt.title("Gráfico")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x,y)
    plt.grid(True)
    plt.axvline(x=0, color='black', linestyle='-')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.show()

# Função que aplica o princípio de bisseção para encontrar a raiz da função em determinado intervalo
# Desenha um gráfico da iteração pela distância em relação ao resultado
def bissecao(a, b, tol, f, grafico_individual=False, grafico_comparativo=False):
    # Criação de variáveis para o relatório
    inicio = a
    fim = b
    lista_iteracao = []
    qtd_iteracao = 0
    lista_resultado = []

    if (f(a) * f(b) < 0):
        while True:
            meio = (a+b)/2
            resultado = f(meio)
            qtd_iteracao += 1
            lista_iteracao.append(qtd_iteracao)
            lista_resultado.append(abs(resultado))
            if (abs(resultado) < tol):
                break
            if resultado > 0:
                b = meio
            else:
                a = meio
        if grafico_individual==False and grafico_comparativo==False:
            print(f"Raiz da Função: {meio}\n")
            print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
            print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
        elif grafico_individual==True:
            print(f"Raiz da Função: {meio}\n")
            print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
            print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
            print("\n\n GRÁFICO DA RELAÇÃO DISTÂNCIA DO RESULTADO DA FUNÇÃO E A QUANTIDADE DE ITERAÇÕES\n\n")
            plt.figure(figsize=(10,6))
            plt.title("Iterações X Distância do Resultado")
            plt.xlabel("Iterações")
            plt.ylabel("Distância do Resultado")
            plt.plot(lista_iteracao, lista_resultado)
            plt.axvline(x=0, color='black', linestyle='-')
            plt.axhline(y=0, color='black', linestyle='-')
            plt.scatter(x=lista_iteracao[-1], y=0, color="red")
            plt.grid(True)
            plt.show()
        else:
            return lista_iteracao, lista_resultado
    else:
        print(f"A raiz da função não está no intervalo informado!")

# Função que retorna a raiz de uma função que foi linearizada com gráfico
def secante (f, a, b, grafico_individual=False, grafico_comparativo=False, tol=1e-6, max_iter=100):
    # Criação de variáveis para o relatório
    inicio = a
    fim = b
    lista_iteracao = []
    qtd_iteracao = 0
    lista_resultado = []

    for i in range(max_iter):
        denominador = f(b) - f(a)
        if abs(denominador) < 1e-14:
            print("Erro: Divisão por zero evitada. f(b) - f(a) ≈ 0")
            return None

        prox_ponto = b - f(b) * (b - a) / denominador
        qtd_iteracao += 1
        lista_iteracao.append(qtd_iteracao)
        lista_resultado.append(abs(f(prox_ponto)))

        if abs(prox_ponto - b) <= tol:
            if grafico_individual==False and grafico_comparativo==False:
                print("RELATÓRIO:\n\n") 
                print(f"Raiz da Função: {prox_ponto}\n")
                print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
                print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
            elif grafico_individual==True:
                print("RELATÓRIO:\n\n") 
                print(f"Raiz da Função: {prox_ponto}\n")
                print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
                print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
                print("\n\n GRÁFICO DA RELAÇÃO DISTÂNCIA DO RESULTADO DA FUNÇÃO E A QUANTIDADE DE ITERAÇÕES\n\n")
                plt.figure(figsize=(10,6))
                plt.title("Iterações X Distância do Resultado")
                plt.xlabel("Iterações")
                plt.ylabel("Distância do Resultado")
                plt.plot(lista_iteracao, lista_resultado)
                plt.axvline(x=0, color='black', linestyle='-')
                plt.axhline(y=0, color='black', linestyle='-')
                plt.scatter(x=lista_iteracao[-1], y=0, color="red")
                plt.grid(True)
                plt.show()
            else:
                return lista_iteracao, lista_resultado

        a = b
        b = prox_ponto
    return None


# Função que retorna a raiz de uma função que foi linearizada (entre intervalo de sinais iguais) com gráfico
def falsa_posicao(f, a, b, grafico_individual=False, grafico_comparativo=False ,tol=1e-6, max_iter=100):
    # Criação de variáveis para o relatório
    inicio = a
    fim = b
    lista_iteracao = []
    qtd_iteracao = 0
    lista_resultado = []
    if f(a) * f(b) >= 0:
        print("ops! f(a) e f(b) devem ter sinais opostos")
        return None
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        qtd_iteracao += 1
        lista_iteracao.append(qtd_iteracao)
        lista_resultado.append(abs(f(c)))
        if abs(f(c)) < tol and i==99:
            if grafico_individual==False and grafico_comparativo==False:
                print("RELATÓRIO:\n\n")
                print(f"Raiz da Função:{c}\n")
                print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
                print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
            elif grafico_individual==True:
                print("RELATÓRIO:\n\n")
                print(f"Raiz da Função: {c}\n")
                print(f"Intervalo: [{inicio:.0f},{fim:.0f}]\n")
                print(f"Quantidade de iterações necessárias: {qtd_iteracao}\n")
                print("\n\n GRÁFICO DA RELAÇÃO DISTÂNCIA DO RESULTADO DA FUNÇÃO E A QUANTIDADE DE ITERAÇÕES\n\n")
                plt.figure(figsize=(10,6))
                plt.title("Iterações X Distância do Resultado")
                plt.xlabel("Iterações")
                plt.ylabel("Distância do Resultado")
                plt.plot(lista_iteracao, lista_resultado)
                plt.axvline(x=0, color='black', linestyle='-')
                plt.axhline(y=0, color='black', linestyle='-')
                plt.scatter(x=lista_iteracao[-1], y=0, color="red")
                plt.grid(True)
                plt.show()
            else:
                return lista_iteracao, lista_resultado
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c   
    return None
    
def grafico_comparativo(f,a,b,tol):
    lista_iteracao_bis, lista_resultado_bis = bissecao(a,b,tol,f,False,True)
    lista_iteracao_sec, lista_resultado_sec = secante(f,a,b,False,True)
    lista_iteracao_falsa_posicao, lista_resultado_falsa_posicao = falsa_posicao(f,a,b,False,True)
    plt.figure(figsize=(10,6))
    plt.title("Comparativo: Iterações X Distância do Resultado")
    plt.xlabel("Iterações")
    plt.ylabel("Distâncias do Resultado")
    plt.plot(lista_iteracao_bis, lista_resultado_bis, label="Bisseção", color="purple")
    plt.plot(lista_iteracao_sec, lista_resultado_sec, label="Secante", color="green")
    plt.plot(lista_iteracao_falsa_posicao, lista_resultado_falsa_posicao, label="Falsa Posição", color="blue")
    plt.axvline(x=0, color='black', linestyle='-')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    plt.yscale("log")
    plt.show()