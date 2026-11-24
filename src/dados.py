"""Coleta dos dados diários do Open-Meteo (reanálise ERA5 e API de previsão)."""
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src import config

URL_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
URL_PREVISAO = "https://api.open-meteo.com/v1/forecast"


def _get(url, params, tentativas=3):
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tentativas - 1:
                raise
            time.sleep(60)   # a API gratuita limita requisições por minuto (HTTP 429)


def _para_tabela(json):
    df = pd.DataFrame(json["daily"]).rename(columns={"time": "data", **config.VARIAVEIS})
    df["data"] = pd.to_datetime(df["data"])
    return df


def baixar_historico(inicio=config.INICIO, fim=None):
    """Série diária da reanálise ERA5. O ERA5 chega com ~6 dias de atraso: a API devolve
    esses últimos dias vazios, e eles são descartados."""
    fim = fim or str(date.today() - timedelta(days=1))
    params = {"latitude": config.LATITUDE, "longitude": config.LONGITUDE, "start_date": inicio,
              "end_date": fim, "daily": ",".join(config.VARIAVEIS), "timezone": config.FUSO,
              "models": "era5_seamless"}
    df = _para_tabela(_get(URL_HISTORICO, params))
    vazio = df[["tmax", "tmin"]].isna().any(axis=1)
    return df[~vazio[::-1].cummin()[::-1]].reset_index(drop=True)   # tira só o "rabo" vazio


def baixar_recentes(dias=14):
    """Últimos dias pela API de previsão (modelos meteorológicos, sem o atraso do ERA5).
    É o que a demonstração usa para prever a máxima de amanhã com o dado de hoje."""
    params = {"latitude": config.LATITUDE, "longitude": config.LONGITUDE, "past_days": dias,
              "forecast_days": 1, "daily": ",".join(config.VARIAVEIS), "timezone": config.FUSO}
    df = _para_tabela(_get(URL_PREVISAO, params))
    return df[df["data"] <= pd.Timestamp(date.today())].reset_index(drop=True)


if __name__ == "__main__":
    df = baixar_historico()
    config.DADOS_BRUTOS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DADOS_BRUTOS, index=False)
    print(f"{len(df)} dias salvos em {config.DADOS_BRUTOS} ({df.data.min():%d/%m/%Y} a {df.data.max():%d/%m/%Y})")
