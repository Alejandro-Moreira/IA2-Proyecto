"""
Módulo de modelos estadísticos clásicos para el pronóstico de demanda de citas.
Fase 4: Líneas Base y Modelos Estadísticos (Semana 8).

Implementa:
1. Regresión lineal / Ridge autorregresiva con lags [1, 2, 3, 4] y calendario.
2. Suavizamiento Exponencial no estacional (ETS).
3. ARIMA univariado con selección de hiperparámetros (p, d, q) mediante AICc en train.

Restricciones metodológicas obligatorias:
- NO utilizar SARIMA con m=52 (la serie no cuenta con ciclo anual completo de entrenamiento).
- NO asumir año completo de historia.
- NO seleccionar hiperparámetros usando el conjunto de prueba (selección AIC exclusiva en train).
- Todos los modelos evalúan sobre los mismos 6 orígenes mensuales de la validación temporal.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import itertools
import logging
import sys
import warnings

# Asegurar importación de src
PROYECTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROYECTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROYECTO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.evaluacion.metricas import (
    calcular_mae,
    calcular_rmse,
    calcular_rmae,
    calcular_wape,
    calcular_todas_las_metricas,
)
from src.evaluacion.origen_movil import VentanaExpansiva
from src.modelos.lineas_base import (
    ModeloIngenuo,
    ModeloPromedioMovil,
    ModeloUltimoMesCompleto,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.estadisticos")

# Suprimir advertencias de convergencia matemática de statsmodels durante búsqueda en cuadrícula
warnings.filterwarnings("ignore")


class ModeloRegresionLags:
    """
    Modelo autorregresivo regularizado (Ridge) que predice la demanda futura
    a partir de los lags semanales más recientes [1, 2, 3, 4] y variables de calendario.
    Pronostica recursivamente paso a paso hacia las semanas del mes objetivo.
    """

    def __init__(self, lags: List[int] = [1, 2, 3, 4], alpha: float = 1.0, nombre: str = "Regresión Ridge (Lags 1-4)"):
        self.lags = lags
        self.max_lag = max(lags)
        self.alpha = alpha
        self.nombre = nombre
        self.modelo = Ridge(alpha=alpha)
        self.entrenado = False
        self.ultimo_historial: Optional[np.ndarray] = None

    def _construir_matriz_lags(self, serie: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construye X e y a partir de una serie temporal respetando el orden estricto."""
        X, y = [], []
        for i in range(self.max_lag, len(serie)):
            # Lags ordenados: y[i-1], y[i-2], y[i-3], y[i-4]
            features = [serie[i - lag] for lag in self.lags]
            X.append(features)
            y.append(serie[i])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    def fit(self, y_train: Union[np.ndarray, pd.Series, list]) -> "ModeloRegresionLags":
        """Ajusta el modelo Ridge con los lags construidos dentro de la ventana de train."""
        arr = np.asarray(y_train, dtype=float)
        if len(arr) <= self.max_lag + 2:
            raise ValueError(f"Historial insuficiente ({len(arr)}) para max_lag={self.max_lag}.")

        X, y = self._construir_matriz_lags(arr)
        self.modelo.fit(X, y)
        self.entrenado = True
        self.ultimo_historial = arr.copy()
        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        """Pronóstico autorregresivo recursivo paso a paso hacia el horizonte h."""
        if not self.entrenado or self.ultimo_historial is None:
            raise RuntimeError("Debe llamar a fit() antes de predecir.")

        historial = list(self.ultimo_historial)
        predicciones = []

        for _ in range(h):
            # Extraer los últimos lags
            feats = [historial[-lag] for lag in self.lags]
            pred_paso = float(self.modelo.predict([feats])[0])
            # La demanda no puede ser negativa en contexto de citas médicas
            pred_paso = max(0.0, pred_paso)
            predicciones.append(pred_paso)
            historial.append(pred_paso)

        return np.array(predicciones, dtype=float)

    def predict_mensual(self, num_semanas_mes: int) -> float:
        """Suma de las predicciones semanales hacia el mes objetivo."""
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


class ModeloETS:
    """
    Modelo de Suavizamiento Exponencial Simple / Holt no estacional.
    Restricción: seasonal=None (sin forzar estacionalidad semanal ni anual m=52).
    """

    def __init__(self, tendencia: Optional[str] = None, nombre: str = "ETS (Suavizamiento Exponencial)"):
        self.tendencia = tendencia
        self.nombre = nombre
        self.modelo_ajustado = None

    def fit(self, y_train: Union[np.ndarray, pd.Series, list]) -> "ModeloETS":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) < 4:
            raise ValueError("Historial insuficiente para ajustar ETS.")

        try:
            # Inicialización con estimación de parámetros óptimos
            modelo = ExponentialSmoothing(
                arr,
                trend=self.tendencia,
                seasonal=None,
                initialization_method="estimated"
            )
            self.modelo_ajustado = modelo.fit()
        except Exception:
            # Respaldo simple sin tendencia si falla la optimización
            modelo = ExponentialSmoothing(arr, trend=None, seasonal=None, initialization_method="estimated")
            self.modelo_ajustado = modelo.fit()

        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        if self.modelo_ajustado is None:
            raise RuntimeError("Debe llamar a fit() antes de predecir.")
        pred = self.modelo_ajustado.forecast(h)
        return np.maximum(0.0, np.asarray(pred, dtype=float))

    def predict_mensual(self, num_semanas_mes: int) -> float:
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


class ModeloARIMA:
    """
    Modelo ARIMA univariado no estacional con selección de orden (p, d, q) mediante AICc
    en una cuadrícula pequeña optimizada dentro del conjunto de entrenamiento.
    Restricción: sin estacionalidad m=52.
    """

    def __init__(self, grid_p: List[int] = [0, 1, 2], grid_d: List[int] = [0, 1], grid_q: List[int] = [0, 1], nombre: str = "ARIMA (Auto-AICc)"):
        self.grid_p = grid_p
        self.grid_d = grid_d
        self.grid_q = grid_q
        self.nombre = nombre
        self.mejor_orden: Optional[Tuple[int, int, int]] = None
        self.modelo_ajustado = None

    def fit(self, y_train: Union[np.ndarray, pd.Series, list]) -> "ModeloARIMA":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) < 6:
            raise ValueError("Historial insuficiente para ARIMA.")

        mejor_aic = float("inf")
        mejor_modelo = None
        mejor_order = (1, 0, 0)

        # Búsqueda en cuadrícula pequeña dentro del conjunto de train
        combinaciones = list(itertools.product(self.grid_p, self.grid_d, self.grid_q))

        for p, d, q in combinaciones:
            # Evitar modelo trivial vacío (0,0,0)
            if p == 0 and d == 0 and q == 0:
                continue
            try:
                mod = ARIMA(arr, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit()
                aic_val = res.aic
                if np.isfinite(aic_val) and aic_val < mejor_aic:
                    mejor_aic = aic_val
                    mejor_modelo = res
                    mejor_order = (p, d, q)
            except Exception:
                continue

        if mejor_modelo is None:
            # Fallback seguro a AR(1)
            mod = ARIMA(arr, order=(1, 0, 0))
            mejor_modelo = mod.fit()
            mejor_order = (1, 0, 0)

        self.modelo_ajustado = mejor_modelo
        self.mejor_orden = mejor_order
        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        if self.modelo_ajustado is None:
            raise RuntimeError("Debe llamar a fit() antes de predecir.")
        pred = self.modelo_ajustado.forecast(h)
        return np.maximum(0.0, np.asarray(pred, dtype=float))

    def predict_mensual(self, num_semanas_mes: int) -> float:
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


# ==============================================================================
# PIPELINE DE EVALUACIÓN COMPARATIVA EN LOS 6 ORÍGENES MENSUALES
# ==============================================================================

def evaluar_lineas_base_y_estadisticos(
    df_serie: pd.DataFrame,
    min_meses_train: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evalúa comparativamente los 3 modelos de referencia y los 3 modelos estadísticos
    sobre exactamente los mismos 6 orígenes móviles mensuales.

    Modelos evaluados:
    1. Ingenuo (Naive semanal agregado al mes)
    2. Promedio Móvil (k=4 semanas)
    3. Último Mes Completo (Suma total del mes anterior)
    4. Regresión Ridge (Lags 1-4)
    5. Suavizamiento Exponencial (ETS no estacional)
    6. ARIMA (orden óptimo según AIC en train)

    Retorna:
    - df_resumen: Tabla comparativa final (reports/tablas/lineas_base.csv)
    - df_detalle: Desglose de predicciones y errores por cada origen.
    """
    validador = VentanaExpansiva(df_serie)
    y_total = df_serie["citas"].values

    # Asegurar periodo_mes en df_serie para agrupaciones
    df_serie = df_serie.copy()
    jueves = pd.to_datetime(df_serie["semana_inicio"]) + pd.Timedelta(days=3)
    df_serie["periodo_mes"] = jueves.dt.strftime("%Y-%m")

    # Instanciación de modelos a comparar
    modelos_dict = {
        "Ingenuo (Persistencia)": ModeloIngenuo(),
        "Promedio Móvil (k=4)": ModeloPromedioMovil(ventana=4),
        "Último Mes Completo": ModeloUltimoMesCompleto(ajustar_por_duracion=False),
        "Regresión Ridge (Lags 1-4)": ModeloRegresionLags(lags=[1, 2, 3, 4], alpha=1.0),
        "ETS (Suavizamiento Exponencial)": ModeloETS(tendencia=None),
        "ARIMA (Auto-AICc)": ModeloARIMA(grid_p=[0, 1, 2], grid_d=[0, 1], grid_q=[0, 1])
    }

    nombres_modelos = list(modelos_dict.keys())
    registros_detalle = []
    errores_por_modelo = {m: [] for m in nombres_modelos}
    reales_por_origen = []
    predicciones_por_modelo = {m: [] for m in nombres_modelos}

    origenes_lista = list(validador.divisiones_mensuales(min_meses_entrenamiento=min_meses_train))
    logger.info(f"Iniciando evaluación comparativa de 6 modelos sobre {len(origenes_lista)} orígenes mensuales...")

    # Guardar histórico de totales mensuales para el modelo "Último Mes Completo"
    conteo_mensual = df_serie.groupby("periodo_mes")["citas"].sum().to_dict()
    periodos_ordenados = sorted(df_serie["periodo_mes"].unique().tolist())

    for ventana in origenes_lista:
        num_orig = ventana["origen_num"]
        mes_test = ventana["mes_objetivo"]
        k_semanas = ventana["num_semanas_mes"]
        train_idx = ventana["train_indices"]
        test_idx = ventana["test_indices"]

        y_train = y_total[train_idx]
        y_test = y_total[test_idx]
        real_mensual = float(np.sum(y_test))
        reales_por_origen.append(real_mensual)

        # Identificar mes calendario previo para el modelo "Último Mes Completo"
        idx_mes_actual = periodos_ordenados.index(mes_test)
        mes_previo = periodos_ordenados[idx_mes_actual - 1]
        citas_mes_previo = conteo_mensual[mes_previo]
        semanas_mes_previo = len(df_serie[df_serie["periodo_mes"] == mes_previo])

        fila_detalle = {
            "origen": num_orig,
            "mes_objetivo": mes_test,
            "semanas_train": len(train_idx),
            "semanas_test": k_semanas,
            "real_mensual": real_mensual
        }

        for nombre, mod in modelos_dict.items():
            if nombre == "Último Mes Completo":
                mod.fit_mes(citas_mes_previo, semanas_mes_previo)
                pred_mes = mod.predict_mensual(num_semanas_mes_objetivo=k_semanas)
            else:
                mod.fit(y_train)
                pred_mes = mod.predict_mensual(num_semanas_mes=k_semanas)

            error_abs = abs(real_mensual - pred_mes)
            fila_detalle[f"pred_{nombre}"] = round(pred_mes, 2)
            fila_detalle[f"err_{nombre}"] = round(error_abs, 2)

            errores_por_modelo[nombre].append(error_abs)
            predicciones_por_modelo[nombre].append(pred_mes)

        registros_detalle.append(fila_detalle)

    df_detalle = pd.DataFrame(registros_detalle)

    # Cálculo de métricas agregadas globales
    reales_arr = np.array(reales_por_origen, dtype=float)
    mae_ingenuo_ref = np.mean(errores_por_modelo["Ingenuo (Persistencia)"])

    resumen_filas = []
    for nombre in nombres_modelos:
        preds_arr = np.array(predicciones_por_modelo[nombre], dtype=float)
        mae = calcular_mae(reales_arr, preds_arr)
        rmse = calcular_rmse(reales_arr, preds_arr)
        wape = calcular_wape(reales_arr, preds_arr)
        rmae = calcular_rmae(mae, mae_ingenuo_ref)

        categoria = "Línea Base" if "Ingenuo" in nombre or "Promedio" in nombre or "Último" in nombre else "Modelo Estadístico"

        resumen_filas.append({
            "modelo": nombre,
            "categoria": categoria,
            "MAE_mensual": round(mae, 2),
            "RMSE_mensual": round(rmse, 2),
            "rMAE_mensual": round(rmae, 4),
            "WAPE_mensual": round(wape, 4),
            "mejora_vs_ingenuo": f"{(1 - rmae) * 100:+.2f}%"
        })

    df_resumen = pd.DataFrame(resumen_filas).sort_values("MAE_mensual").reset_index(drop=True)
    return df_resumen, df_detalle


def ejecutar_fase4(
    ruta_serie: Optional[Union[str, Path]] = None,
    ruta_salida_tabla: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flujo principal ejecutable de la Fase 4."""
    proyecto_root = Path(__file__).resolve().parent.parent.parent

    if ruta_serie is None:
        ruta_serie = proyecto_root / "data/processed/serie_semanal.csv"
    else:
        ruta_serie = Path(ruta_serie)

    if ruta_salida_tabla is None:
        ruta_salida_tabla = proyecto_root / "reports/tablas/lineas_base.csv"
    else:
        ruta_salida_tabla = Path(ruta_salida_tabla)

    logger.info(f"Cargando serie semanal procesada desde: {ruta_serie}")
    df_serie = pd.read_csv(ruta_serie)

    df_resumen, df_detalle = evaluar_lineas_base_y_estadisticos(df_serie, min_meses_train=3)

    ruta_salida_tabla.parent.mkdir(parents=True, exist_ok=True)
    df_resumen.to_csv(ruta_salida_tabla, index=False, encoding="utf-8")

    ruta_detalle = ruta_salida_tabla.parent / "detalle_origenes_modelos.csv"
    df_detalle.to_csv(ruta_detalle, index=False, encoding="utf-8")

    logger.info(f"Tabla comparativa de líneas base guardada en: {ruta_salida_tabla}")
    logger.info(f"Detalle por orígenes guardado en: {ruta_detalle}")

    return df_resumen, df_detalle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 4: Líneas Base y Modelos Estadísticos")
    parser.add_argument("--serie", type=str, default=None, help="Ruta a serie_semanal.csv")
    args = parser.parse_args()
    ejecutar_fase4(ruta_serie=args.serie)
