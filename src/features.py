"""Limpeza, atributos e divisão temporal.

Cada linha representa um dia t e usa só informação disponível até o fim de t
para prever a temperatura máxima do dia t+1. Nada olha para o futuro."""
import numpy as np
import pandas as pd

from src import config

# Faixas fisicamente possíveis em Jaraguá do Sul. Fora disso é erro de dado, não clima extremo.
FAIXAS = {"tmax": (-5, 45), "tmin": (-10, 35), "tmed": (-8, 40), "umidade": (0, 100), "orvalho": (-15, 32),
          "pressao": (960, 1050), "nuvens": (0, 100), "vento_max": (0, 150), "rajada": (0, 200),
          "vento_dir": (0, 360), "chuva": (0, 400), "horas_chuva": (0, 24), "radiacao": (0, 40),
          "evapotranspiracao": (0, 15)}


def carregar(caminho=config.DADOS_BRUTOS):
    return pd.read_csv(caminho, parse_dates=["data"])


def diagnostico(df):
    """Por coluna: ausentes, valores fora da faixa física e outliers (z robusto acima de 5)."""
    linhas = {}
    for col, (minimo, maximo) in FAIXAS.items():
        s = df[col]
        desvio = (s - s.median()).abs()
        escala = 1.4826 * desvio.median()    # chuva tem mediana 0: sem escala, sem z
        z = desvio / escala if escala > 0 else desvio * 0
        linhas[col] = {"ausentes": int(s.isna().sum()),
                       "fora_da_faixa": int((~s.between(minimo, maximo) & s.notna()).sum()),
                       "outliers_z5": int((z > 5).sum()), "min": s.min(), "max": s.max()}
    return pd.DataFrame.from_dict(linhas, orient="index")


def limpar(df):
    """Datas únicas e em ordem, calendário diário completo (dia faltando vira linha vazia, para os
    lags continuarem sendo "o dia anterior") e valor fisicamente impossível vira ausente,
    imputado depois pela mediana do treino dentro do Pipeline."""
    df = df.drop_duplicates("data", keep="last").set_index("data").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), name="data"))
    for col, (minimo, maximo) in FAIXAS.items():
        if col in df:
            df[col] = df[col].where(df[col].between(minimo, maximo))
    df.loc[df["tmin"] > df["tmax"], ["tmin", "tmax"]] = np.nan
    return df


def construir_atributos(df):
    """Atributos do dia t e o alvo (máxima de t+1). A última linha fica sem alvo: é a da previsão."""
    f = df[list(FAIXAS)].copy()
    for lag in (1, 2, 3):
        f[f"tmax_lag{lag}"] = f["tmax"].shift(lag)
    f["tmax_media3"] = f["tmax"].rolling(3).mean()
    f["tmax_media7"] = f["tmax"].rolling(7).mean()
    f["tmax_delta"] = f["tmax"].diff()
    f["amplitude"] = f["tmax"] - f["tmin"]
    # Pressão caindo costuma anunciar frente fria (e queda de temperatura) no Sul do Brasil.
    f["pressao_delta1"] = f["pressao"].diff(1)
    f["pressao_delta2"] = f["pressao"].diff(2)
    f["umidade_delta"] = f["umidade"].diff()
    f["chuva_3d"] = f["chuva"].rolling(3).sum()
    # Depressão do ponto de orvalho: ar seco (valor alto) esquenta mais durante o dia.
    f["deficit_orvalho"] = f["tmed"] - f["orvalho"]
    # Direção é circular (350° e 10° são vizinhos), então vira seno e cosseno.
    direcao = np.deg2rad(f.pop("vento_dir"))
    f["vento_dir_sin"], f["vento_dir_cos"] = np.sin(direcao), np.cos(direcao)
    # Estação do ano, também circular.
    dia = 2 * np.pi * f.index.day_of_year / 365.25
    f["dia_ano_sin"], f["dia_ano_cos"] = np.sin(dia), np.cos(dia)
    f["alvo"] = f["tmax"].shift(-1)
    return f.iloc[6:]                                   # a primeira semana não tem média de 7 dias


def lista_atributos(dataset):
    """Colunas candidatas a entrada do modelo (tudo menos o alvo)."""
    return [c for c in dataset.columns if c != "alvo"]


def dividir(dataset):
    """Divisão cronológica: treino | validação | teste. Linhas sem alvo ficam de fora."""
    rotulado = dataset[dataset["alvo"].notna()]
    um_dia = pd.Timedelta(days=1)
    treino = rotulado.loc[: config.FIM_TREINO]
    validacao = rotulado.loc[pd.Timestamp(config.FIM_TREINO) + um_dia: config.FIM_VALIDACAO]
    teste = rotulado.loc[pd.Timestamp(config.FIM_VALIDACAO) + um_dia:]
    return treino, validacao, teste


def preparar(df_bruto):
    """Atalho: dado bruto -> dataset com atributos."""
    return construir_atributos(limpar(df_bruto))
