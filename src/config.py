"""Configurações compartilhadas pelo projeto: local, variáveis, divisão temporal e caminhos."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS_BRUTOS = RAIZ / "data" / "raw" / "jaragua_diario.csv"
MODELO = RAIZ / "models" / "modelo_tmax.joblib"
FIGURAS = RAIZ / "reports" / "figuras"

CIDADE = "Jaraguá do Sul (SC)"
LATITUDE = -26.48611
LONGITUDE = -49.06667
FUSO = "America/Sao_Paulo"
INICIO = "2010-01-01"
SEMENTE = 42

# nome na API do Open-Meteo -> nome no projeto
VARIAVEIS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "temperature_2m_mean": "tmed",
    "relative_humidity_2m_mean": "umidade",
    "dew_point_2m_mean": "orvalho",
    "pressure_msl_mean": "pressao",
    "cloud_cover_mean": "nuvens",
    "wind_speed_10m_max": "vento_max",
    "wind_gusts_10m_max": "rajada",
    "wind_direction_10m_dominant": "vento_dir",
    "precipitation_sum": "chuva",
    "precipitation_hours": "horas_chuva",
    "shortwave_radiation_sum": "radiacao",
    "et0_fao_evapotranspiration": "evapotranspiracao",
}

# Divisão cronológica (o futuro nunca entra no treino).
FIM_TREINO = "2022-12-31"      # 2010-2022: treino (13 anos)
FIM_VALIDACAO = "2024-12-31"   # 2023-2024: validação (seleção de atributos conferida, escolha do modelo)
                               # 2025 em diante: teste, usado uma única vez no fim

# Seleção de atributos por votação: cada método vota nas suas K melhores e entra quem tiver VOTOS_MINIMOS votos.
VOTACAO_K = 10
VOTOS_MINIMOS = 2
MIN_ATRIBUTOS = 6
