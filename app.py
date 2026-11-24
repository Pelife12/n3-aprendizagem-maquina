"""Interface simples da demonstração: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src import config, dados, previsao

st.set_page_config(page_title="Máxima de amanhã · Jaraguá do Sul", page_icon="🌡️", layout="centered")


@st.cache_data(ttl=3600, show_spinner="Buscando os últimos dias no Open-Meteo...")
def dias_recentes():
    return dados.baixar_recentes()


@st.cache_data
def dias_exemplo():
    return pd.read_csv(config.RAIZ / "data" / "exemplo_entrada.csv", parse_dates=["data"])


def mostrar(resultado):
    r = resultado
    st.markdown(f"## 🌡️ Máxima prevista para {r['data_alvo']:%d/%m/%Y}: **{r['previsto']:.1f} °C**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Máxima prevista", f"{r['previsto']:.1f} °C", f"{r['previsto'] - r['tmax_hoje']:+.1f} °C em relação a hoje")
    col2.metric("Faixa provável", f"{r['faixa'][0]:.0f} a {r['faixa'][1]:.0f} °C", help="8 em cada 10 dias a máxima real cai aqui")
    col3.metric("Máxima de hoje", f"{r['tmax_hoje']:.1f} °C")
    st.markdown("**Por quê:**\n" + "\n".join(f"- {s}" for s in r["sinais"]))
    st.caption(f"Dados até {r['data_base']:%d/%m/%Y}. Modelo {r['modelo']} treinado até {r['treinado_ate']}.")


st.title("🌡️ Qual será a máxima amanhã em Jaraguá do Sul?")
st.write("Modelo de aprendizagem de máquina treinado com 15 anos de dados diários do ERA5 (Open-Meteo). "
         "Ele olha as condições de hoje e dos últimos dias e estima a temperatura máxima de amanhã.")

aba_hoje, aba_simulador, aba_modelo = st.tabs(["Previsão de hoje", "Simulador “e se...?”", "Sobre o modelo"])

with aba_hoje:
    try:
        recentes = dias_recentes()
        mostrar(previsao.prever(recentes))
        with st.expander("Dados usados (API de previsão do Open-Meteo)"):
            st.dataframe(recentes.set_index("data").tail(7), use_container_width=True)
    except Exception as erro:  # sem internet ou API fora do ar: a demonstração segue com o exemplo
        st.warning(f"Não foi possível buscar os dados de hoje ({erro}). Mostrando o arquivo de exemplo.")
        mostrar(previsao.prever(dias_exemplo()))

with aba_simulador:
    st.write("Parte de uma semana real (arquivo de exemplo) e deixa você mudar as condições do **último dia**. "
             "Assim dá para ver como cada sinal mexe na máxima prevista para o dia seguinte.")
    base = dias_exemplo()
    ultimo = base.iloc[-1]
    c1, c2 = st.columns(2)
    tmax = c1.slider("Máxima de hoje (°C)", 8.0, 40.0, float(ultimo.tmax), 0.5)
    tmin = c2.slider("Mínima de hoje (°C)", 0.0, 28.0, float(min(ultimo.tmin, tmax)), 0.5)
    variacao = c1.slider("Variação da pressão desde ontem (hPa)", -10.0, 10.0, 0.0, 0.5,
                         help="Negativo = pressão caindo (frente fria chegando). Positivo = pressão subindo.")
    nuvens = c2.slider("Cobertura de nuvens hoje (%)", 0, 100, int(ultimo.nuvens))
    vento = c1.selectbox("Direção do vento hoje", ["norte", "nordeste", "leste", "sudeste", "sul", "sudoeste", "oeste",
                                                   "noroeste"], index=2)
    mes = c2.selectbox("Mês", range(1, 13), index=int(ultimo.data.month) - 1,
                       format_func=lambda m: ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set",
                                              "out", "nov", "dez"][m - 1])

    simulado = base.copy()
    simulado["data"] = pd.date_range(end=pd.Timestamp(year=ultimo.data.year, month=mes, day=min(ultimo.data.day, 28)),
                                     periods=len(simulado))
    i, anterior = simulado.index[-1], simulado.index[-2]
    tmin = min(tmin, tmax)
    simulado.loc[i, ["tmax", "tmin", "tmed", "nuvens"]] = [tmax, tmin, (tmax + tmin) / 2, nuvens]
    simulado.loc[i, "pressao"] = simulado.loc[anterior, "pressao"] + variacao
    simulado.loc[i, "vento_dir"] = {"norte": 0, "nordeste": 45, "leste": 90, "sudeste": 135, "sul": 180,
                                    "sudoeste": 225, "oeste": 270, "noroeste": 315}[vento]
    mostrar(previsao.prever(simulado))

with aba_modelo:
    pacote = previsao.carregar_modelo()
    st.markdown(f"""
- **Modelo:** {pacote['nome']}, escolhido entre Ridge, Random Forest e HistGradientBoosting (com todos os atributos ou só
  com os eleitos pela votação) pelo menor erro médio na validação (2023–2024).
- **Treino final:** 2010 até {pacote['treinado_ate']}. **Teste:** 2025 em diante, nunca visto no treino.
- **Faixa provável:** de {pacote['faixa_80'][0]:+.1f} a {pacote['faixa_80'][1]:+.1f} °C em torno da previsão, a partir dos
  erros na validação.
""")
    caminho = config.RAIZ / "reports" / "metricas_teste.json"
    if caminho.exists():
        m = json.loads(caminho.read_text(encoding="utf-8"))
        st.write(f"Resultado no teste ({m['periodo'][0]} a {m['periodo'][1]}, {m['dias']} dias):")
        tabela = pd.DataFrame(m["metricas"]).T[["mae", "rmse", "r2", "erro_ate_2c"]]
        tabela.columns = ["erro médio (°C)", "RMSE (°C)", "R²", "dias com erro ≤ 2 °C"]
        st.dataframe(tabela.round(3), use_container_width=True)
    for figura in ["03_avaliacao.png", "03_importancia.png"]:
        if (config.FIGURAS / figura).exists():
            st.image(str(config.FIGURAS / figura))
    st.caption("Dados: Open-Meteo.com (CC BY 4.0), reanálise ERA5 do Copernicus Climate Change Service.")
