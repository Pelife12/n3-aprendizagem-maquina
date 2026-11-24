"""Testes rápidos das regras que, se quebradas, invalidariam o resultado: vazamento do futuro,
divisão temporal, seleção de atributos e o uso do modelo salvo."""
import numpy as np
import pandas as pd
import pytest

from src import config, features, modelagem, previsao, selecao


@pytest.fixture(scope="module")
def dataset():
    return features.preparar(features.carregar())


def test_alvo_e_a_maxima_do_dia_seguinte(dataset):
    rotulado = dataset[dataset.alvo.notna()]
    assert (rotulado.alvo.iloc[:-1].to_numpy() == rotulado.tmax.iloc[1:].to_numpy()).all()


def test_atributos_nao_usam_o_futuro():
    """Mudar a máxima de amanhã não pode mudar nenhum atributo de hoje."""
    bruto = features.carregar().iloc[:400].copy()
    base = features.preparar(bruto)
    dia = base.index[200]
    alterado = bruto.copy()
    alterado.loc[alterado.data == dia + pd.Timedelta(days=1), "tmax"] += 8
    novo = features.preparar(alterado)
    colunas = features.lista_atributos(base)
    pd.testing.assert_series_equal(base.loc[dia, colunas], novo.loc[dia, colunas])


def test_divisao_e_cronologica_e_sem_sobreposicao(dataset):
    treino, validacao, teste = features.dividir(dataset)
    assert treino.index.max() < validacao.index.min() <= validacao.index.max() < teste.index.min()
    assert treino.index.max() == pd.Timestamp(config.FIM_TREINO)


def test_limpar_troca_valor_impossivel_por_ausente():
    df = pd.DataFrame({"data": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                       **{c: [10.0, 10.0] for c in features.FAIXAS}})
    df.loc[0, "umidade"] = 150
    limpo = features.limpar(df)
    assert len(limpo) == 3                       # o dia 02/01 que faltava vira linha vazia
    assert np.isnan(limpo.loc["2020-01-01", "umidade"])


def test_votacao_por_maioria():
    """Com k=2: a tem 3 votos, b tem 2, os outros 1 ou 0. Com min_atributos=4 completa pelo ranking médio."""
    t = pd.DataFrame({"rank_kbest": [1, 2, 3, 4, 5], "rank_rf": [2, 1, 5, 3, 4], "rank_rfe": [1, 4, 2, 5, 3]},
                     index=list("abcde"), dtype=float)
    assert selecao.votar(t, k=2, votos_minimos=2, min_atributos=2)[1] == ["a", "b"]
    eleitas = selecao.votar(t, k=2, votos_minimos=2, min_atributos=4)[1]
    assert eleitas[:2] == ["a", "b"] and len(eleitas) == 4


def test_metricas_e_faixa():
    m = modelagem.metricas([20, 22, 24], [21, 22, 26])
    assert m["mae"] == pytest.approx(1.0) and m["erro_ate_2c"] == 1.0
    baixo, alto = modelagem.intervalo(np.arange(-5, 6), 0.8)
    assert baixo < 0 < alto


def test_previsao_com_arquivo_de_exemplo():
    dias = pd.read_csv(config.RAIZ / "data" / "exemplo_entrada.csv", parse_dates=["data"])
    r = previsao.prever(dias)
    assert 0 < r["previsto"] < 45
    assert r["faixa"][0] < r["previsto"] < r["faixa"][1]
    assert r["data_alvo"] == (dias.data.max() + pd.Timedelta(days=1)).date()
    with pytest.raises(ValueError):
        previsao.prever(dias.tail(3))
