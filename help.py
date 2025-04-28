def bissecao(f, inicio, final, tolerancia=0.0009):
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