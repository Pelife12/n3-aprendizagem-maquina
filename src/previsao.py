"""Uso do modelo treinado: recebe os últimos dias de dados e devolve a máxima prevista para amanhã."""
from functools import lru_cache

import joblib
import pandas as pd

from src import config, features

DIAS_NECESSARIOS = 7   # a média de 7 dias precisa da semana que termina no dia da previsão


@lru_cache(maxsize=1)
def carregar_modelo():
    """Pacote salvo pelo notebook 02: modelo (Pipeline), atributos, faixa de erro e metadados."""
    return joblib.load(config.MODELO)


def linha_do_dia(dias):
    """Atributos do último dia da tabela, que é o "hoje" da previsão."""
    dias = dias.copy()
    dias["data"] = pd.to_datetime(dias["data"])
    if dias["data"].nunique() < DIAS_NECESSARIOS:
        raise ValueError(f"são necessários pelo menos {DIAS_NECESSARIOS} dias seguidos de dados")
    return features.preparar(dias).tail(1)


def sinais(linha, previsto):
    """Frases curtas sobre as condições de hoje que ajudam a entender a previsão."""
    r = linha.iloc[0]
    frases = [f"máxima de hoje: {r['tmax']:.1f} °C (média dos últimos 7 dias: {r['tmax_media7']:.1f} °C)"]
    if r["pressao_delta2"] <= -3:
        frases.append(f"pressão caindo ({r['pressao_delta2']:+.1f} hPa em 2 dias): costuma anteceder a chegada de frente fria")
    elif r["pressao_delta2"] >= 3:
        frases.append(f"pressão subindo ({r['pressao_delta2']:+.1f} hPa em 2 dias): tempo firme, tendência de aquecimento")
    if r["nuvens"] >= 90:
        frases.append(f"céu muito encoberto hoje ({r['nuvens']:.0f}%): se abrir amanhã, a máxima sobe")
    if r["chuva"] >= 10:
        frases.append(f"choveu {r['chuva']:.0f} mm hoje")
    variacao = previsto - r["tmax"]
    if abs(variacao) >= 2:
        frases.append(f"o modelo espera {'aquecimento' if variacao > 0 else 'resfriamento'} de {abs(variacao):.1f} °C")
    else:
        frases.append("o modelo espera uma máxima parecida com a de hoje")
    return frases


def prever(dias):
    """dias: tabela com a coluna `data` e as variáveis de config.VARIAVEIS (nomes do projeto),
    pelo menos 7 dias seguidos. Devolve um dicionário com a máxima prevista, a faixa e a explicação."""
    pacote = carregar_modelo()
    linha = linha_do_dia(dias)
    previsto = float(pacote["modelo"].predict(linha[pacote["atributos"]])[0])
    baixo, alto = pacote["faixa_80"]
    hoje = linha.index[0]
    return {"data_base": hoje.date(), "data_alvo": (hoje + pd.Timedelta(days=1)).date(),
            "previsto": previsto, "faixa": (previsto + baixo, previsto + alto), "tmax_hoje": float(linha["tmax"].iloc[0]),
            "sinais": sinais(linha, previsto), "modelo": pacote["nome"], "treinado_ate": pacote["treinado_ate"]}


def texto(resultado):
    r = resultado
    linhas = [f"Previsão para {r['data_alvo']:%d/%m/%Y} em {config.CIDADE} (com os dados de {r['data_base']:%d/%m/%Y})",
              f"  Máxima prevista: {r['previsto']:.1f} °C",
              f"  Faixa provável (8 em cada 10 dias): {r['faixa'][0]:.1f} a {r['faixa'][1]:.1f} °C",
              "  Por quê:"]
    linhas += [f"    • {s}" for s in r["sinais"]]
    linhas.append(f"  (modelo {r['modelo']} treinado até {r['treinado_ate']})")
    return "\n".join(linhas)
