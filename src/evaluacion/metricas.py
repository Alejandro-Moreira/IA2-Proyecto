"""
Módulo de métricas de evaluación para series temporales de citas médicas.
Fase 3: Métricas y Validación (Semana 8).

Implementa las métricas de evaluación requeridas:
- MAE: Mean Absolute Error (escala original en citas)
- RMSE: Root Mean Squared Error (penalización cuadrática)
- rMAE: MAE relativo al pronóstico ingenuo (métrica relativa principal en escala mensual)
- WAPE: Weighted Absolute Percentage Error (alternativa robusta porcentual)

Fórmulas matemáticas:
- MAE  = (1/N) * sum(|y_i - y_hat_i|)
- RMSE = sqrt((1/N) * sum((y_i - y_hat_i)^2))
- rMAE = MAE_modelo / MAE_ingenuo
"""

from typing import Dict, Optional, Union
import numpy as np


def calcular_mae(y_real: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calcula el Error Absoluto Medio (MAE).

    Parámetros:
        y_real: Valores reales observados.
        y_pred: Valores predichos por el modelo.

    Retorna:
        float: MAE en las mismas unidades de la serie (citas).
    """
    y_r = np.asarray(y_real, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_r) == 0:
        raise ValueError("Los vectores de entrada no pueden estar vacíos.")
    if len(y_r) != len(y_p):
        raise ValueError(f"Longitudes distintas: y_real ({len(y_r)}) vs y_pred ({len(y_p)}).")

    return float(np.mean(np.abs(y_r - y_p)))


def calcular_rmse(y_real: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE).

    Parámetros:
        y_real: Valores reales observados.
        y_pred: Valores predichos por el modelo.

    Retorna:
        float: RMSE en las mismas unidades de la serie (citas).
    """
    y_r = np.asarray(y_real, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_r) == 0:
        raise ValueError("Los vectores de entrada no pueden estar vacíos.")
    if len(y_r) != len(y_p):
        raise ValueError(f"Longitudes distintas: y_real ({len(y_r)}) vs y_pred ({len(y_p)}).")

    return float(np.sqrt(np.mean((y_r - y_p) ** 2)))


def calcular_rmae(
    mae_modelo: float,
    mae_ingenuo: float,
    epsilon: float = 1e-8
) -> float:
    """
    Calcula el MAE Relativo (rMAE) frente al pronóstico ingenuo de referencia.

    Fórmula:
        rMAE = MAE_modelo / MAE_ingenuo

    Interpretación:
        - rMAE < 1.0: El modelo supera a la línea base ingenua (reducción porcentual del error).
        - rMAE = 1.0: Desempeño idéntico a repetir el valor pasado.
        - rMAE > 1.0: El modelo es inferior a la referencia simple.
    """
    if mae_ingenuo < epsilon:
        # Si el modelo ingenuo tiene error casi cero (predicción perfecta exacta)
        return 1.0 if mae_modelo < epsilon else float("inf")
    return float(mae_modelo / mae_ingenuo)


def calcular_wape(y_real: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calcula el Error Porcentual Absoluto Ponderado (WAPE).
    WAPE = sum(|y - y_hat|) / sum(y)
    Es robusto frente a valores cero individuales en la serie.
    """
    y_r = np.asarray(y_real, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    suma_real = np.sum(y_r)
    if suma_real == 0:
        return 0.0 if np.sum(np.abs(y_r - y_p)) == 0 else float("inf")
    return float(np.sum(np.abs(y_r - y_p)) / suma_real)


def calcular_todas_las_metricas(
    y_real: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_ingenuo: Optional[Union[np.ndarray, list]] = None
) -> Dict[str, float]:
    """
    Calcula el compendio completo de métricas obligatorias para un conjunto de predicciones.

    Retorna un diccionario con:
        - 'MAE': Error absoluto medio
        - 'RMSE': Raíz del error cuadrático medio
        - 'WAPE': Error porcentual absoluto ponderado
        - 'MAE_ingenuo': MAE de la referencia ingenua (si se provee)
        - 'rMAE': MAE relativo (si se provee y_ingenuo)
    """
    mae = calcular_mae(y_real, y_pred)
    rmse = calcular_rmse(y_real, y_pred)
    wape = calcular_wape(y_real, y_pred)

    resultado = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4)
    }

    if y_ingenuo is not None:
        mae_base = calcular_mae(y_real, y_ingenuo)
        rmae = calcular_rmae(mae, mae_base)
        resultado["MAE_ingenuo"] = round(mae_base, 4)
        resultado["rMAE"] = round(rmae, 4)

    return resultado
