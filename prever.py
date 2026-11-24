"""Demonstração por linha de comando: qual será a temperatura máxima amanhã em Jaraguá do Sul?

Uso:
    python prever.py                                       # dados reais dos últimos dias (API do Open-Meteo)
    python prever.py --arquivo data/exemplo_entrada.csv    # a partir de um CSV (funciona sem internet)
    python prever.py --arquivo data/exemplo_entrada.csv --tmax-hoje 33 --nuvens 10   # "e se...?"
"""
import argparse

import pandas as pd

from src import dados, previsao


def main():
    p = argparse.ArgumentParser(description="Temperatura máxima de amanhã em Jaraguá do Sul (SC).")
    p.add_argument("--arquivo", help="CSV com pelo menos 7 dias seguidos (mesmas colunas de data/raw/jaragua_diario.csv)")
    p.add_argument("--tmax-hoje", type=float, help="substitui a máxima (°C) do último dia")
    p.add_argument("--tmin-hoje", type=float, help="substitui a mínima (°C) do último dia")
    p.add_argument("--nuvens", type=float, help="substitui a cobertura de nuvens (%%) do último dia")
    p.add_argument("--chuva", type=float, help="substitui a chuva (mm) do último dia")
    p.add_argument("--pressao-variacao", type=float, help="variação da pressão (hPa) do penúltimo para o último dia")
    args = p.parse_args()

    if args.arquivo:
        dias = pd.read_csv(args.arquivo, parse_dates=["data"])
    else:
        print("Buscando os últimos dias na API do Open-Meteo...")
        dias = dados.baixar_recentes()

    ultimo, penultimo = dias.index[-1], dias.index[-2]
    for coluna, valor in [("tmax", args.tmax_hoje), ("tmin", args.tmin_hoje), ("nuvens", args.nuvens),
                          ("chuva", args.chuva)]:
        if valor is not None:
            dias.loc[ultimo, coluna] = valor
    if args.tmax_hoje is not None or args.tmin_hoje is not None:
        dias.loc[ultimo, "tmed"] = (dias.loc[ultimo, "tmax"] + dias.loc[ultimo, "tmin"]) / 2
    if args.pressao_variacao is not None:
        dias.loc[ultimo, "pressao"] = dias.loc[penultimo, "pressao"] + args.pressao_variacao

    print(previsao.texto(previsao.prever(dias)))


if __name__ == "__main__":
    main()
