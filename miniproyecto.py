"""Miniproyecto 1: la serie del grupo, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. Las
partes marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""

import itertools
import warnings

import holidays
import matplotlib

# Se utiliza un backend no interactivo para generar las figuras sin abrir ventanas.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)

# Se suprimen las advertencias para mantener limpia la salida durante la ejecución.
warnings.filterwarnings("ignore")

# Se definen las rutas de salida y se crean las carpetas si aún no existen.
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ============================================================
# 1. La serie
# ============================================================

# Se especifican el archivo de entrada, las columnas y la periodicidad mensual.
ARCHIVO = "datos/cemento_colombia.csv"
FECHA = "fecha"
VALOR = "despachos"
PERIODO = 12
ESCALA = 1000 # los modelos se ajustan en miles de toneladas

# Se documenta la decisión metodológica sobre el tratamiento de la serie.
DECISION = "Sin valores nulos. Se decide NO recortar la serie y mantener el histórico completo (incluyendo 2020)."


def preparar(serie):
    """Aplica la decisión sobre huecos y atípicos: la serie pasa completa."""
    # Se identifica el mes con el menor registro, sin eliminar observaciones.
    print(f"Mes con menor valor (caída atípica): {serie.idxmin():%Y-%m} con {serie.min():.2f} ton")
    return serie


def graficar_serie(serie):
    """Grafica la serie que entra a los modelos."""
    # Se representa la evolución temporal de los despachos y se guarda la figura.
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.set_title("Despachos de Cemento Gris (Serie Completa)")
    eje.set_ylabel("Toneladas")
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close("all")


# ============================================================
# 2. Componentes
# ============================================================

def descomponer(serie):
    """STL robusta con periodo 12: guarda figuras/stl.png e imprime su lectura."""
    # La descomposición STL separa tendencia, estacionalidad y residuo.
    ajuste_stl = STL(serie, period=PERIODO, robust=True).fit()

    # Se guarda la representación gráfica de los componentes estimados.
    figura = ajuste_stl.plot()
    figura.set_size_inches(10, 7)
    figura.suptitle("Descomposición STL (Robusta) - Despachos de Cemento", y=1.02, fontsize=14)
    figura.savefig(FIGURAS / "stl.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # Se extraen los componentes para resumir su comportamiento.
    tendencia = ajuste_stl.trend
    estacional = ajuste_stl.seasonal.groupby(ajuste_stl.seasonal.index.month).mean()
    meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
             7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
    max_res_date = ajuste_stl.resid.abs().idxmax()
    max_res_val = ajuste_stl.resid.loc[max_res_date]

    # Se presentan los valores extremos de la tendencia, la estacionalidad y los residuos.
    print(f"Tendencia: {tendencia.iloc[0]:,.0f} ton en {tendencia.index[0]:%Y-%m} "
          f"a {tendencia.iloc[-1]:,.0f} ton en {tendencia.index[-1]:%Y-%m}")
    print(f"Mes con estacionalidad más ALTA: {meses[estacional.idxmax()]} ({estacional.max():.2f})")
    print(f"Mes con estacionalidad más BAJA: {meses[estacional.idxmin()]} ({estacional.min():.2f})")
    print(f"Mes con el residuo más GRANDE: {max_res_date:%Y-%m} (Valor: {max_res_val:.2f})")
    return ajuste_stl


# ============================================================
# 3. Estacionariedad e identificación
# ============================================================

def diferenciar(serie, d, D):
    """Aplica d diferencias regulares y D estacionales a la serie."""
    # Se aplican primero las diferencias estacionales y luego las regulares.
    serie_dif = serie.copy()
    if D > 0:
        serie_dif = serie_dif.diff(PERIODO * D)
    if d > 0:
        serie_dif = serie_dif.diff(d)
    return serie_dif


def estacionariedad(serie):
    """ADF sobre la serie y sus tres diferencias; elige (d, D) y guarda figuras/acf_pacf.png."""
    # Se preparan cuatro versiones para evaluar la estacionariedad mediante ADF.
    candidatas = [
        (0, 0, "Original (d=0, D=0)", serie),
        (1, 0, "Dif. regular (d=1, D=0)", serie.diff(1)),
        (0, 1, "Dif. estacional (d=0, D=1)", serie.diff(PERIODO)),
        (1, 1, "Dif. reg+est (d=1, D=1)", serie.diff(PERIODO).diff(1)),
    ]

    resultados_adf = []
    estacionarias = []
    for d, D, nombre, s in candidatas:
        # Se ajusta la especificación de la prueba según la versión evaluada.
        reg = "ct" if d == 0 and D == 0 else "c"
        estadistico, p_valor, rezagos, n_obs, _, _ = adfuller(s.dropna(), regression=reg, autolag="AIC")
        resultados_adf.append({"Versión": nombre, "Regresión": reg,
                               "Estadístico ADF": estadistico, "Valor p": p_valor,
                               "Rezagos usados": rezagos, "Observaciones": n_obs})
        # La tupla (d + D, d, D, ...) ordena por menos diferencias y, a igual número, por menor d
        if p_valor < 0.05:
            estacionarias.append((d + D, d, D, nombre, p_valor))

    # Se muestran los resultados de la prueba para las cuatro versiones de la serie.
    print("Prueba ADF:")
    print(pd.DataFrame(resultados_adf).to_string(index=False))

    # Se selecciona la combinación de diferencias con menor orden entre las estacionarias.
    if estacionarias:
        _, d_elegido, D_elegido, nombre_elegido, p_elegido = sorted(estacionarias)[0]
        print(f"\n-> Elección AUTOMÁTICA por Parsimonia: d={d_elegido}, D={D_elegido} ({nombre_elegido}); p-valor={p_elegido:.4f}")
    else:
        d_elegido, D_elegido = 1, 1
        print("\n-> Ninguna versión obtuvo p < 0.05. Se fuerza d=1, D=1 por defecto.")

    # Se transforma la serie con el orden de diferenciación seleccionado.
    serie_estacionaria = diferenciar(serie, d_elegido, D_elegido).dropna()

    # Se identifican los rezagos significativos de las funciones ACF y PACF.
    _, conf_acf = acf(serie_estacionaria, nlags=36, alpha=0.05, fft=False)
    _, conf_pacf = pacf(serie_estacionaria, nlags=36, method="ywm", alpha=0.05)
    rezagos_acf = [k for k in range(1, 37) if conf_acf[k, 0] > 0 or conf_acf[k, 1] < 0]
    rezagos_pacf = [k for k in range(1, 37) if conf_pacf[k, 0] > 0 or conf_pacf[k, 1] < 0]
    print("Rezagos que salen de la banda ACF  (q, Q): ", rezagos_acf)
    print("Rezagos que salen de la banda PACF (p, P):", rezagos_pacf)

    # Se grafican las funciones ACF y PACF para apoyar la identificación de los órdenes del modelo.
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.4))
    plot_acf(serie_estacionaria, lags=36, ax=ejes[0], zero=False)
    ejes[0].set_title(f"ACF (d={d_elegido}, D={D_elegido})")
    plot_pacf(serie_estacionaria, lags=36, method="ywm", ax=ejes[1], zero=False)
    ejes[1].set_title(f"PACF (d={d_elegido}, D={D_elegido})")
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    return d_elegido, D_elegido


# ============================================================
# 4. Modelos clásicos
# ============================================================

# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

def elegir_sarima(serie, d, D):
    """Rejilla (p, q, P, Q) en {0, 1} por AIC y Ljung-Box de los residuos del elegido.

    Guarda figuras/residuos.png y devuelve el orden y el orden estacional.
    """
    # Se escala la serie para facilitar el ajuste de los modelos.
    serie_escalada = serie / ESCALA
    resultados = []

    # Se prueban las 16 combinaciones posibles de órdenes no estacionales y estacionales.
    for p, q, P, Q in itertools.product([0, 1], repeat=4):
        try:
            modelo = ARIMA(serie_escalada, order=(p, d, q), seasonal_order=(P, D, Q, PERIODO)).fit()
            resultados.append({"order": (p, d, q), "seasonal_order": (P, D, Q, PERIODO), "AIC": modelo.aic})
        except Exception:
            continue

    # Se ordenan los modelos según el AIC para identificar los de mejor ajuste relativo.
    df_rejilla = pd.DataFrame(resultados).sort_values("AIC").reset_index(drop=True)
    print("Tabla de AIC (Top 5 mejores modelos):")
    print(df_rejilla.head(5).to_string(index=False))

    # Se selecciona la combinación con el menor AIC.
    mejor_orden = df_rejilla.loc[0, "order"]
    mejor_estacional = df_rejilla.loc[0, "seasonal_order"]
    print(f"\n-> Modelo Elegido: SARIMA {mejor_orden} x {mejor_estacional}")

    # Se reajusta el modelo seleccionado para analizar sus residuos.
    ajuste = ARIMA(serie_escalada, order=mejor_orden, seasonal_order=mejor_estacional).fit()

    # Los primeros d + 12 D residuos son de arranque, no errores de pronóstico
    residuos = ajuste.resid.iloc[d + PERIODO * D:]

    # Se aplica Ljung-Box para evaluar si los residuos conservan autocorrelación.
    lb = acorr_ljungbox(residuos, lags=[3, 12, 24])
    print("\nPrueba de Ljung-Box sobre los residuos (H0: Ruido blanco):")
    print(lb.round(4).to_string())

    # Se representa la evolución de los residuos y su autocorrelación.
    figura, ejes = plt.subplots(2, 1, figsize=(10, 7))
    ejes[0].plot(residuos.index, residuos.values)
    ejes[0].set_title(f"Residuos del SARIMA {mejor_orden}x{mejor_estacional}")
    plot_acf(residuos, lags=24, ax=ejes[1], zero=False)
    ejes[1].set_title("ACF de los residuos")
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    return mejor_orden, mejor_estacional


def modelo_sarima(orden, orden_estacional):
    """Devuelve la función modelo(entrenamiento, h) del SARIMA elegido."""
    # Se construye una función compatible con el procedimiento común de backtesting.
    def modelo(entrenamiento, h):
        ajuste = ARIMA(entrenamiento / ESCALA, order=orden, seasonal_order=orden_estacional).fit()
        return ajuste.forecast(steps=h).to_numpy() * ESCALA
    return modelo


def exogenas(serie):
    """Festivos de Colombia por mes, con el índice de la serie y los 12 meses siguientes."""
    # Se genera el calendario de festivos para cubrir el histórico y el horizonte de pronóstico.
    anios = range(serie.index[0].year, serie.index[-1].year + 2)
    fechas = pd.date_range(start=serie.index[0], periods=len(serie) + 12, freq="MS")

    # Se cuenta el número de festivos de Colombia en cada mes.
    calendario = holidays.country_holidays("CO", years=anios)
    fechas_festivos = pd.to_datetime(list(calendario.keys()))
    conteo = pd.Series(1, index=fechas_festivos).resample("MS").sum()

    # Se alinean los conteos con las fechas de la serie y se asigna cero a los meses sin festivos.
    return conteo.reindex(fechas, fill_value=0).to_frame(name="festivos")


def modelo_sarimax(orden, orden_estacional, x):
    """Devuelve modelo(entrenamiento, h) de un SARIMAX con las exógenas x.

    Ajusta con x.loc[entrenamiento.index] y pronostica con los festivos del
    calendario de los h meses siguientes, que se conocen de antemano.
    """
    # Se construye una función que incorpora las variables exógenas al ajuste y al pronóstico.
    def modelo(entrenamiento, h):
        x_train = x.loc[entrenamiento.index]
        x_test = x.loc[fechas_siguientes(entrenamiento, h)]
        ajuste = ARIMA(entrenamiento / ESCALA, order=orden, seasonal_order=orden_estacional,
                       exog=x_train).fit()
        return ajuste.forecast(steps=h, exog=x_test).to_numpy() * ESCALA
    return modelo


# ============================================================
# 5. La tabla de métricas
# ============================================================

def fila(nombre, serie, modelo):
    _, promedio = backtest(serie, modelo)
    return {"modelo": nombre, "decision": DECISION,
            "RMSE": round(promedio["RMSE"], 2), "MAE": round(promedio["MAE"], 2),
            "MAPE": round(promedio["MAPE"], 2)}


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    # Se carga la serie y se informa su extensión y cantidad de valores faltantes.
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")

    # Se prepara la serie y se genera su gráfico inicial.
    serie = preparar(cruda)
    graficar_serie(serie)

    # Se calculan las métricas de referencia con los modelos ingenuo y estacional ingenuo.
    filas = [fila("ingenuo", serie, ingenuo),
             fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]

    # Se ejecuta la descomposición; si está pendiente, se informa y continúa.
    try:
        descomponer(serie)
    except NotImplementedError as error:
        pendiente("parte 2", error)

    # Se evalúa la estacionariedad y se obtienen los órdenes de diferenciación.
    try:
        d, D = estacionariedad(serie)
    except NotImplementedError as error:
        pendiente("parte 3", error)
        d = D = None

    # Si se determinaron los órdenes de diferenciación, se ajustan y evalúan los modelos.
    if d is not None:
        try:
            orden, estacional = elegir_sarima(serie, d, D)
            filas.append(fila(f"SARIMA{orden}{estacional}", serie,
                              modelo_sarima(orden, estacional)))
            x = exogenas(serie)
            filas.append(fila(f"SARIMAX{orden}{estacional} + exógenas", serie,
                              modelo_sarimax(orden, estacional, x)))
        except NotImplementedError as error:
            pendiente("parte 4", error)

    # Se consolida la comparación de modelos y se guarda la tabla de métricas.
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])


if __name__ == "__main__":
    main()