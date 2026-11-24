"""Seleção de atributos por votação de três métodos diferentes, sempre ajustados só com o treino."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def tres_metodos(treino, candidatas, k=config.VOTACAO_K, semente=config.SEMENTE):
    """Ranking de cada método:
    - SelectKBest (teste F): relação linear de cada atributo, sozinho, com o alvo;
    - Random Forest: importância em um modelo não linear que vê os atributos juntos;
    - RFE com Ridge: elimina, um a um, o atributo de menor peso num modelo linear."""
    X = Pipeline([("imputacao", SimpleImputer(strategy="median")),
                  ("escala", StandardScaler())]).fit_transform(treino[candidatas])
    y = treino["alvo"]
    kbest = SelectKBest(f_regression, k=k).fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=semente, n_jobs=-1).fit(X, y)
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=k).fit(X, y)

    tab = pd.DataFrame({"f_score": kbest.scores_, "rf_importancia": rf.feature_importances_,
                        "rfe_ranking": rfe.ranking_}, index=candidatas)
    tab["rank_kbest"] = tab["f_score"].rank(ascending=False, method="first")
    tab["rank_rf"] = tab["rf_importancia"].rank(ascending=False, method="first")
    # O RFE dá ranking 1 para as k que sobraram; entre elas desempata o |coeficiente| final do Ridge.
    tab["rfe_coef"] = 0.0
    tab.loc[rfe.support_, "rfe_coef"] = abs(rfe.estimator_.coef_)
    ordem = tab.sort_values(["rfe_ranking", "rfe_coef"], ascending=[True, False]).index
    tab["rank_rfe"] = pd.Series(range(1, len(ordem) + 1), index=ordem, dtype=float)
    return tab


def votar(tab, k=config.VOTACAO_K, votos_minimos=config.VOTOS_MINIMOS, min_atributos=config.MIN_ATRIBUTOS):
    """Cada método vota nas suas k melhores; entra quem tiver pelo menos `votos_minimos` votos.
    Se a maioria eleger menos que `min_atributos`, completa pelo menor ranking médio."""
    tab = tab.copy()
    for metodo in ("kbest", "rf", "rfe"):
        tab[f"voto_{metodo}"] = tab[f"rank_{metodo}"] <= k
    tab["votos"] = tab[["voto_kbest", "voto_rf", "voto_rfe"]].sum(axis=1)
    tab["ranking_medio"] = tab[["rank_kbest", "rank_rf", "rank_rfe"]].mean(axis=1)
    tab = tab.sort_values(["votos", "ranking_medio"], ascending=[False, True])
    eleitas = tab.index[tab["votos"] >= votos_minimos].tolist()
    if len(eleitas) < min_atributos:
        eleitas = tab.index[:min_atributos].tolist()
    tab["eleita"] = tab.index.isin(eleitas)
    return tab, eleitas
