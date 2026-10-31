"""Miniproyecto 1: la serie del grupo, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. Las
partes marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)

warnings.filterwarnings("ignore")
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ============================================================
# 1. La serie
# ============================================================
ARCHIVO = "datos/cemento_colombia.csv"
FECHA = "fecha"
VALOR = "despachos"
PERIODO = 12

DECISION = "Sin valores nulos. Se decide NO recortar la serie y mantener el histórico completo (incluyendo 2020), aislando los atípicos mediante STL robusta."

def preparar(serie):
    """Devuelve la serie completa sin modificaciones, ya que no hay NaN 
    y se decidió no recortar los atípicos. Adicionalmente, imprime 
    un resumen exploratorio en consola.
    """
    # --- Evidencia del Análisis Exploratorio en Consola ---
    print("\n" + "="*50)
    print(" ANÁLISIS EXPLORATORIO DE LA SERIE")
    print("="*50)
    print(f"Total de observaciones: {len(serie)}")
    print(f"Rango temporal: {serie.index.min():%Y-%m} a {serie.index.max():%Y-%m}")
    print(f"Valores nulos (NaN): {serie.isna().sum()}")
    print(f"Mes con menor valor (caída atípica): {serie.idxmin():%Y-%m} con {serie.min():.2f} ton")
    print("="*50 + "\n")
    
    return serie


def graficar_serie(serie):
    """Grafica la serie que entra a los modelos"""
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.set_title("Despachos de Cemento Gris (Serie Completa)")
    eje.set_ylabel("Toneladas")
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close('all')
    
# ============================================================
# 2. Componentes
# ============================================================
from statsmodels.tsa.seasonal import STL

def descomponer(serie):
    """Aplica STL robusta a la serie y grafica la descomposición.

    El periodo es PERIODO (12).
    Esta función calcula la descomposición, imprime el análisis
    para el informe y guarda la gráfica 'stl.png' en FIGURAS.
    """
    # 1. Se aplica STL robusta con periodo=12 (PERIODO global)
    ajuste_stl = STL(serie, period=PERIODO, robust=True).fit()
    
    # 2. Guardamos la figura de los componentes STL
    figura = plt.figure(figsize=(10, 7))
    ajuste_stl.plot()
    plt.suptitle("Descomposición STL (Robusta) - Despachos de Cemento", y=1.02, fontsize=14)
    plt.savefig(FIGURAS / "stl.png", dpi=120, bbox_inches='tight')
    plt.close('all') # Cerramos para liberar memoria
    
    # 3. --- Evidencia en Consola para el Informe ---
    print("\n" + "="*50)
    print(" LECTURA DE COMPONENTES STL (PARA EL INFORME)")
    print("="*50)
    
    estacional = ajuste_stl.seasonal.groupby(ajuste_stl.seasonal.index.month).mean()
    meses = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 
             7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}
    
    print(f"Mes con estacionalidad más ALTA: {meses[estacional.idxmax()]} ({estacional.max():.2f})")
    print(f"Mes con estacionalidad más BAJA: {meses[estacional.idxmin()]} ({estacional.min():.2f})")
    
    residuos_abs = ajuste_stl.resid.abs()
    max_res_date = residuos_abs.idxmax()
    max_res_val = ajuste_stl.resid.loc[max_res_date]
    
    print(f"Mes con el residuo más GRANDE: {max_res_date:%Y-%m} (Valor: {max_res_val:.2f})")
    print("="*50 + "\n")
    
    # Retornamos el ajuste para no romper el flujo si el main lo necesita internamente
    return ajuste_stl

# ============================================================
# 3. Estacionariedad e identificación
# ============================================================
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def diferenciar(serie, d, D):
    """Aplica d diferencias regulares y D estacionales a la serie."""
    serie_dif = serie.copy()
    if D > 0:
        serie_dif = serie_dif.diff(PERIODO * D)
    if d > 0:
        serie_dif = serie_dif.diff(d)
    return serie_dif


def estacionariedad(serie):
    """ADF sobre la serie, su diferencia regular, la estacional y las dos.
    Imprima el valor p de cada versión, elija d y D automáticamente, y guarde 
    la ACF y la PACF de la serie elegida en figuras/acf_pacf.png. Devuelva (d, D).
    """
    print("\n" + "="*70)
    print(" 3. ESTACIONARIEDAD E IDENTIFICACIÓN (ADF)")
    print("="*70)
    
    # 1. Definimos las 4 combinaciones candidatas
    candidatas = [
        (0, 0, "Original (d=0, D=0)", serie),
        (1, 0, "Dif. regular (d=1, D=0)", serie.diff(1)),
        (0, 1, "Dif. estacional (d=0, D=1)", serie.diff(PERIODO)),
        (1, 1, "Dif. reg+est (d=1, D=1)", serie.diff(PERIODO).diff(1))
    ]
    
    resultados_adf = []
    estacionarias = []
    
    # 2. Iteramos y calculamos ADF para cada versión
    for d, D, nombre, s in candidatas:
        s_limpia = s.dropna()
        # Regresión "ct" para la original (0,0), "c" para el resto
        reg = "ct" if d == 0 and D == 0 else "c"
        
        estadistico, p_valor, rezagos, n_obs, valores_criticos, icbest = adfuller(
            s_limpia, regression=reg, autolag="AIC"
        )
        
        resultados_adf.append({
            "Versión": nombre,
            "Regresión": reg,
            "Estadístico ADF": estadistico,
            "Valor p": p_valor,
            "Rezagos usados": rezagos,
            "Observaciones": n_obs
        })
        
        # Guardamos las que pasaron la prueba (p < 0.05)
        # La tupla (d+D, d, D, ...) permite que la función sorted() ordene primero
        # por la menor cantidad total de diferencias (parsimonia). 
        # Si empatan en cantidad, el menor valor de 'd' irá primero.
        if p_valor < 0.05:
            estacionarias.append((d + D, d, D, nombre, p_valor))
            
    # Imprimimos la tabla completa
    tabla_adf = pd.DataFrame(resultados_adf)
    print("\nTabla de Resultados Prueba ADF:")
    print(tabla_adf.to_string(index=False))
    
    # 3. Selección automática de d y D
    if estacionarias:
        # sorted() ordena automáticamente de menor a mayor basado en el primer elemento de la tupla (d+D)
        _, d_elegido, D_elegido, nombre_elegido, p_elegido = sorted(estacionarias)[0]
        print(f"\n-> Elección AUTOMÁTICA por Parsimonia: d={d_elegido}, D={D_elegido} ({nombre_elegido}); p-valor={p_elegido:.4f}")
    else:
        # Salvaguarda: si ninguna pasa, forzamos d=1, D=1 por defecto
        d_elegido, D_elegido = 1, 1
        print("\n-> Ninguna versión obtuvo p < 0.05. Se fuerza d=1, D=1 por defecto.")
    
    # 4. Generamos la serie estacionaria final usando la elección del algoritmo
    serie_estacionaria = diferenciar(serie, d_elegido, D_elegido).dropna()
    
    # 5. Extraemos los rezagos que se salen de las bandas
    val_acf, conf_acf = acf(serie_estacionaria, nlags=36, alpha=0.05, fft=False)
    val_pacf, conf_pacf = pacf(serie_estacionaria, nlags=36, method="ywm", alpha=0.05)
    
    rezagos_acf = [k for k in range(1, 37) if conf_acf[k, 0] > 0 or conf_acf[k, 1] < 0]
    rezagos_pacf = [k for k in range(1, 37) if conf_pacf[k, 0] > 0 or conf_pacf[k, 1] < 0]
    
    print("\nRezagos que salen de la banda ACF  (q, Q): ", rezagos_acf)
    print("Rezagos que salen de la banda PACF (p, P):", rezagos_pacf)
    print("="*70 + "\n")
    
    # 6. Graficamos y guardamos la ACF y PACF
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.4))
    plot_acf(serie_estacionaria, lags=36, ax=ejes[0], zero=False)
    ejes[0].set_title(f"ACF (d={d_elegido}, D={D_elegido})")
    
    plot_pacf(serie_estacionaria, lags=36, method="ywm", ax=ejes[1], zero=False)
    ejes[1].set_title(f"PACF (d={d_elegido}, D={D_elegido})")
    
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120, bbox_inches='tight')
    plt.close('all')
    
    return d_elegido, D_elegido

# ============================================================
# 4. Modelos clásicos
# ============================================================
# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import holidays

def elegir_sarima(serie, d, D):
    """Rejilla pequeña de órdenes (p, q, P, Q) comparada por AIC.

    Ajuste con toda la serie preparada, imprima la tabla de AIC, revise los
    residuos del elegido con Ljung-Box y guarde figuras/residuos.png.
    Devuelva el orden y el orden estacional elegidos.
    """
    print("\n" + "="*70)
    print(" 4. SELECCIÓN DEL SARIMA Y DIAGNÓSTICO DE RESIDUOS")
    print("="*70)
    
    # Escalamos la serie para que el optimizador no falle por varianza alta
    serie_escalada = serie / 1000.0
    resultados = []
    
    # Rejilla pequeña: iteramos p, q, P, Q en {0, 1}
    for p, q, P, Q in itertools.product([0, 1], repeat=4):
        try:
            # Notar que usamos los parámetros d y D que recibimos de la Parte 3
            modelo = ARIMA(serie_escalada, order=(p, d, q), seasonal_order=(P, D, Q, PERIODO)).fit()
            resultados.append({
                "order": (p, d, q),
                "seasonal_order": (P, D, Q, PERIODO),
                "AIC": modelo.aic
            })
        except Exception:
            continue
            
    # Ordenamos por AIC (el menor es el mejor)
    df_rejilla = pd.DataFrame(resultados).sort_values("AIC").reset_index(drop=True)
    
    print("Tabla de AIC (Top 5 mejores modelos):")
    print(df_rejilla.head(5).to_string(index=False))
    
    mejor_orden = df_rejilla.loc[0, "order"]
    mejor_estacional = df_rejilla.loc[0, "seasonal_order"]
    print(f"\n-> Modelo Elegido: SARIMA {mejor_orden} x {mejor_estacional}")
    
    # Ajustamos el modelo ganador para revisar sus residuos
    ajuste = ARIMA(serie_escalada, order=mejor_orden, seasonal_order=mejor_estacional).fit()
    
    # Descartamos los primeros meses (d + D*PERIODO) por ser de arranque
    residuos = ajuste.resid.iloc[d + PERIODO * D :]
    
    # Prueba de Ljung-Box a los rezagos 12 y 24
    lb = acorr_ljungbox(residuos, lags=[12, 24])
    print("\nPrueba de Ljung-Box sobre los residuos (H0: Ruido blanco):")
    print(lb.round(4).to_string())
    
    # Generamos la gráfica de residuos en el tiempo y su ACF
    figura, ejes = plt.subplots(2, 1, figsize=(10, 7))
    ejes[0].plot(residuos.index, residuos.values)
    ejes[0].set_title(f"Residuos del SARIMA {mejor_orden}x{mejor_estacional}")
    
    plot_acf(residuos, lags=24, ax=ejes[1], zero=False)
    ejes[1].set_title("ACF de los residuos")
    
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120, bbox_inches='tight')
    plt.close('all')
    
    print("="*70 + "\n")
    return mejor_orden, mejor_estacional


def modelo_sarima(orden, orden_estacional):
    """Devuelve la función modelo(entrenamiento, h) del SARIMA elegido."""
    def modelo(entrenamiento, h):
        # Ajustamos el modelo escalando el entrenamiento local
        ajuste = ARIMA(entrenamiento / 1000.0, order=orden, seasonal_order=orden_estacional).fit()
        # Pronosticamos y des-escalamos (* 1000)
        return ajuste.forecast(steps=h).to_numpy() * 1000.0
    return modelo


def exogenas(serie):
    """DataFrame de variables conocidas de antemano, con el índice de la serie.

    Por ejemplo, el número de festivos o de días hábiles de cada mes. Debe
    cubrir también los 12 meses que siguen al final de la serie.
    """
    # Años a cubrir (desde el inicio hasta un año posterior al fin de la serie)
    anios = range(serie.index[0].year, serie.index[-1].year + 2)
    # Fechas completas (incluyendo el horizonte de pronóstico +12 meses)
    fechas = pd.date_range(start=serie.index[0], periods=len(serie) + 12, freq='MS')
    
    # Extraemos festivos de Colombia
    cal_colombia = holidays.country_holidays('CO', years=anios)
    fechas_festivos = pd.to_datetime(list(cal_colombia.keys()))
    
    # Contamos la cantidad de festivos por cada mes y lo pasamos a un DataFrame
    conteo = pd.Series(1, index=fechas_festivos).resample('MS').sum()
    df_exog = conteo.reindex(fechas, fill_value=0).to_frame(name="festivos")
    
    return df_exog


def modelo_sarimax(orden, orden_estacional, x):
    """Devuelve modelo(entrenamiento, h) de un SARIMAX con exógenas x.

    Use x.loc[entrenamiento.index] para ajustar y
    x.loc[fechas_siguientes(entrenamiento, h)] para pronosticar.
    """
    def modelo(entrenamiento, h):
        # Seleccionamos exógenas para el periodo de entrenamiento
        x_train = x.loc[entrenamiento.index]
        
        ajuste = ARIMA(
            entrenamiento / 1000.0, 
            order=orden, 
            seasonal_order=orden_estacional, 
            exog=x_train
        ).fit()
        
        # Obtenemos las fechas del futuro usando la función de comun.py
        fechas_futuras = fechas_siguientes(entrenamiento, h)
        x_test = x.loc[fechas_futuras]
        
        return ajuste.forecast(steps=h, exog=x_test).to_numpy() * 1000.0
        
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
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    filas = [fila("ingenuo", serie, ingenuo),
             fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]

    try:
        descomponer(serie)
    except NotImplementedError as error:
        pendiente("parte 2", error)

    try:
        d, D = estacionariedad(serie)
    except NotImplementedError as error:
        pendiente("parte 3", error)
        d = D = None

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

    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])


if __name__ == "__main__":
    main()
