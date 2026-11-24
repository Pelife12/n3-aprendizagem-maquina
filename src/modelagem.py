"""Modelos candidatos, referências ingênuas e métricas de regressão."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def candidatos():
    """Três famílias diferentes: uma linear (interpretável), um bagging e um boosting de árvores.
    A imputação (e a escala, no Ridge) ficam dentro do Pipeline: são ajustadas só no treino."""
    return {
        "Ridge": Pipeline([
            ("imputacao", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
            ("modelo", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("imputacao", SimpleImputer(strategy="median")),
            ("modelo", RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                             n_jobs=-1, random_state=config.SEMENTE)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputacao", SimpleImputer(strategy="median")),
            ("modelo", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                     min_samples_leaf=20, l2_regularization=1.0,
                                                     random_state=config.SEMENTE)),
        ]),
    }


# Grades pequenas de hiperparâmetros, buscadas com validação cruzada temporal só no treino.
GRADES = {
    "Ridge": {"modelo__alpha": [0.1, 1.0, 10.0, 100.0]},
    "Random Forest": {"modelo__min_samples_leaf": [5, 10, 25], "modelo__max_depth": [None, 12]},
    "HistGradientBoosting": {"modelo__learning_rate": [0.03, 0.05, 0.1], "modelo__max_leaf_nodes": [7, 15, 31]},
}


class Persistencia:
    """Referência 1: a máxima de amanhã repete a de hoje."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["tmax"].to_numpy(dtype=float)


class Climatologia:
    """Referência 2: a máxima média daquela época do ano (média móvel de 15 dias do dia do ano) no treino."""

    def fit(self, X, y):
        media = pd.Series(np.asarray(y), index=X.index).groupby(X.index.day_of_year).mean()
        media = media.reindex(range(1, 367)).interpolate(limit_direction="both")
        estendida = pd.concat([media.iloc[-7:], media, media.iloc[:7]])
        self.media_ = estendida.rolling(15, center=True).mean().iloc[7:-7]
        return self

    def predict(self, X):
        # o alvo é o dia seguinte, então usa o dia do ano de t+1
        dia_alvo = (X.index + pd.Timedelta(days=1)).day_of_year
        return self.media_.reindex(dia_alvo).to_numpy(dtype=float)


def metricas(real, previsto):
    real, previsto = np.asarray(real, dtype=float), np.asarray(previsto, dtype=float)
    erro = np.abs(real - previsto)
    return {"mae": float(mean_absolute_error(real, previsto)),
            "rmse": float(root_mean_squared_error(real, previsto)),
            "r2": float(r2_score(real, previsto)),
            "erro_ate_2c": float((erro <= 2).mean()),       # fração de dias com erro de no máximo 2 °C
            "erro_max": float(erro.max())}


def intervalo(residuos, cobertura=0.8):
    """Faixa de erro empírica: quantis dos resíduos (real - previsto) da validação.
    Com cobertura 0,8, cerca de 8 em cada 10 dias a máxima real cai dentro da faixa."""
    cauda = (1 - cobertura) / 2
    return float(np.quantile(residuos, cauda)), float(np.quantile(residuos, 1 - cauda))
