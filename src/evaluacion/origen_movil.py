"""
Módulo de validación temporal mediante esquema de origen móvil (Ventana Expansiva).
Fase 3: Métricas y Validación (Semana 8).

Implementa:
1. Ventana expansiva mensual (Resultado Principal):
   - El punto de corte es el cierre de cada mes.
   - Pronostica las semanas del mes siguiente.
   - Agrega semanalmente para obtener la estimación mensual: Y_hat_mes = sum(y_hat_semanal).
   - Compara contra el total real mensual: Y_real_mes = sum(y_real_semanal).
   - Reporta rMAE mensual frente al pronóstico ingenuo de referencia.
   - Documenta de forma transparente la limitación de disponer de aproximadamente 6 orígenes mensuales.
2. Ventana expansiva semanal (Complementaria):
   - Evalúa horizontes semanales individuales h = 1, 2, 3, 4, 5.
   - Reporta métricas desagregadas por horizonte.
3. Transformaciones sin fuga temporal (Data Leakage Free):
   - Escaladores y normalizaciones ajustados (fit) exclusivamente con datos de entrenamiento.
"""

from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import argparse
import logging
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
PROYECTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROYECTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROYECTO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.evaluacion.metricas import (
    calcular_mae,
    calcular_rmse,
    calcular_rmae,
    calcular_todas_las_metricas,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.origen_movil")


class EscaladorTemporal:
    """
    Escalador estándar (Z-score) diseñado para evitar estrictamente la fuga temporal.
    Garantiza que fit() solo se ejecute con datos del conjunto de entrenamiento.
    """

    def __init__(self):
        self.media: Optional[float] = None
        self.desv: Optional[float] = None
        self.ajustado: bool = False

    def fit(self, x_train: Union[np.ndarray, pd.Series, list]) -> "EscaladorTemporal":
        """Ajusta media y desviación estándar únicamente con el historial pasado de entrenamiento."""
        arr = np.asarray(x_train, dtype=float)
        if len(arr) == 0:
            raise ValueError("No se puede ajustar el escalador con un conjunto vacío.")
        self.media = float(np.mean(arr))
        self.desv = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        if self.desv < 1e-8:
            self.desv = 1.0
        self.ajustado = True
        return self

    def transform(self, x: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """Aplica la transformación z = (x - mu) / sigma usando parámetros aprendidos en train."""
        if not self.ajustado:
            raise RuntimeError("El escalador debe ser ajustado con fit() antes de transform().")
        arr = np.asarray(x, dtype=float)
        return (arr - self.media) / self.desv

    def fit_transform(self, x_train: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """Ajusta y transforma sobre datos de entrenamiento."""
        return self.fit(x_train).transform(x_train)

    def inverse_transform(self, x_norm: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """Deshace la normalización: x = z * sigma + mu."""
        if not self.ajustado:
            raise RuntimeError("El escalador debe ser ajustado con fit() antes de inverse_transform().")
        arr = np.asarray(x_norm, dtype=float)
        return arr * self.desv + self.media


class VentanaExpansiva:
    """
    Gestor de particiones temporales cronológicas mediante ventana expansiva.
    Garantiza que para cualquier origen t, el entrenamiento contenga solo índices <= t,
    y la evaluación contenga exclusivamente índices futuros > t.
    """

    def __init__(self, df_serie: pd.DataFrame):
        self.df = df_serie.copy().reset_index(drop=True)
        self.df["semana_inicio"] = pd.to_datetime(self.df["semana_inicio"])
        # Cálculo del año y mes objetivo según regla del jueves
        self.df["jueves"] = self.df["semana_inicio"] + pd.Timedelta(days=3)
        self.df["anio"] = self.df["jueves"].dt.year
        self.df["mes"] = self.df["jueves"].dt.month
        self.df["periodo_mes"] = self.df["jueves"].dt.strftime("%Y-%m")

    def divisiones_semanales(
        self,
        min_entrenamiento: int = 16,
        horizontes: List[int] = [1, 2, 3, 4, 5]
    ) -> Generator[Dict[str, Union[int, List[int]]], None, None]:
        """
        Genera particiones para la evaluación semanal complementaria.
        Para cada origen t (índice de corte):
        - train_indices: [0, 1, ..., t]
        - test_indices_por_horizonte: {h: [t + h]} para h en horizontes válidos.
        """
        max_h = max(horizontes)
        n_total = len(self.df)

        for t in range(min_entrenamiento - 1, n_total - max_h):
            train_idx = list(range(0, t + 1))
            test_por_h = {h: t + h for h in horizontes if (t + h) < n_total}

            # Validación de seguridad contra fuga temporal
            assert max(train_idx) < min(test_por_h.values()), "Fuga temporal en división semanal."

            yield {
                "origen_t": t,
                "semana_origen": self.df.loc[t, "semana_inicio"].strftime("%Y-%m-%d"),
                "train_indices": train_idx,
                "test_por_horizonte": test_por_h
            }

    def divisiones_mensuales(
        self,
        min_meses_entrenamiento: int = 3
    ) -> Generator[Dict[str, Union[str, int, List[int]]], None, None]:
        """
        Genera particiones para la evaluación mensual principal (agregación de semanas).
        Identifica los meses calendario completos.
        Para cada mes de prueba:
        - train_indices: todas las semanas de meses estrictamente anteriores.
        - test_indices: todas las semanas pertenecientes al mes objetivo.
        """
        periodos_ordenados = self.df["periodo_mes"].unique().tolist()

        # Solo meses con al menos 4 semanas completas para evaluación
        conteo_semanas_por_mes = self.df.groupby("periodo_mes").size().to_dict()
        meses_evaluables = [
            m for m in periodos_ordenados if conteo_semanas_por_mes.get(m, 0) >= 4
        ]

        logger.info(
            f"Meses completos identificados ({len(meses_evaluables)}): {meses_evaluables}. "
            f"Min meses train: {min_meses_entrenamiento}."
        )

        for idx_mes_test in range(min_meses_entrenamiento, len(meses_evaluables)):
            mes_test = meses_evaluables[idx_mes_test]
            meses_train = meses_evaluables[:idx_mes_test]

            train_idx = self.df[self.df["periodo_mes"].isin(meses_train)].index.tolist()
            test_idx = self.df[self.df["periodo_mes"] == mes_test].index.tolist()

            # Validación de no solapamiento ni inversión temporal
            assert len(train_idx) > 0, "Conjunto de entrenamiento vacío."
            assert len(test_idx) > 0, "Conjunto de prueba vacío."
            assert max(train_idx) < min(test_idx), (
                f"Violación de orden cronológico en mes {mes_test}: "
                f"max_train={max(train_idx)} >= min_test={min(test_idx)}"
            )
            assert set(train_idx).isdisjoint(set(test_idx)), "Solapamiento entre train y test."

            yield {
                "origen_num": idx_mes_test - min_meses_entrenamiento + 1,
                "mes_objetivo": mes_test,
                "num_semanas_mes": len(test_idx),
                "train_indices": train_idx,
                "test_indices": test_idx,
                "semanas_train_total": len(train_idx)
            }


# ==============================================================================
# MODELOS DE LÍNEA BASE PARA EVALUACIÓN EN ORIGEN MÓVIL
# ==============================================================================

def predecir_ingenuo_semanal(y_train: np.ndarray, h: int) -> np.ndarray:
    """Pronóstico ingenuo (Naive): repite el último valor disponible del historial."""
    ultimo_valor = y_train[-1]
    return np.full(shape=h, fill_value=ultimo_valor, dtype=float)


def predecir_promedio_movil(y_train: np.ndarray, h: int, ventana: int = 4) -> np.ndarray:
    """Pronóstico por Promedio Móvil de las últimas k semanas."""
    k = min(ventana, len(y_train))
    promedio = float(np.mean(y_train[-k:]))
    return np.full(shape=h, fill_value=promedio, dtype=float)


def ejecutar_evaluacion_mensual_origenes(
    df_serie: pd.DataFrame,
    min_meses_train: int = 3
) -> pd.DataFrame:
    """
    Ejecuta la evaluación mensual principal de origen móvil comparando:
    1. Pronóstico Ingenuo (Naive semanal agregado al mes)
    2. Promedio Móvil (Moving Average k=4 semanas agregado al mes)

    Calcula errores mensuales agregados y rMAE para cada origen.
    """
    validador = VentanaExpansiva(df_serie)
    filas_resultados = []

    y_total_serie = df_serie["citas"].values

    for ventana in validador.divisiones_mensuales(min_meses_entrenamiento=min_meses_train):
        train_idx = ventana["train_indices"]
        test_idx = ventana["test_indices"]
        mes_obj = ventana["mes_objetivo"]
        k_semanas = ventana["num_semanas_mes"]

        y_train = y_total_serie[train_idx]
        y_test_semanal = y_total_serie[test_idx]

        # Total real mensual observado
        total_real_mes = float(np.sum(y_test_semanal))

        # 1. Predicciones semanales del modelo Ingenuo y agregación mensual
        pred_ingenuo_sem = predecir_ingenuo_semanal(y_train, h=k_semanas)
        total_pred_ingenuo = float(np.sum(pred_ingenuo_sem))

        # 2. Predicciones semanales del Promedio Móvil (k=4) y agregación mensual
        pred_pm_sem = predecir_promedio_movil(y_train, h=k_semanas, ventana=4)
        total_pred_pm = float(np.sum(pred_pm_sem))

        error_abs_ingenuo = abs(total_real_mes - total_pred_ingenuo)
        error_abs_pm = abs(total_real_mes - total_pred_pm)

        filas_resultados.append({
            "origen": ventana["origen_num"],
            "mes_objetivo": mes_obj,
            "semanas_entrenamiento": len(train_idx),
            "semanas_mes_test": k_semanas,
            "real_mensual": total_real_mes,
            "pred_ingenuo_mensual": total_pred_ingenuo,
            "error_abs_ingenuo": error_abs_ingenuo,
            "pred_pm4_mensual": total_pred_pm,
            "error_abs_pm4": error_abs_pm
        })

    df_res = pd.DataFrame(filas_resultados)

    # Cálculo del resumen global mensual
    if len(df_res) > 0:
        mae_ingenuo_global = df_res["error_abs_ingenuo"].mean()
        mae_pm_global = df_res["error_abs_pm4"].mean()
        rmae_pm = calcular_rmae(mae_pm_global, mae_ingenuo_global)

        logger.info(f"Evaluación Mensual completada sobre {len(df_res)} orígenes móviles.")
        logger.info(f"MAE Mensual Ingenuo: {mae_ingenuo_global:.2f} citas")
        logger.info(f"MAE Mensual Promedio Móvil: {mae_pm_global:.2f} citas (rMAE: {rmae_pm:.4f})")
    else:
        logger.warning("No se generaron orígenes mensuales con los datos disponibles.")

    return df_res


def ejecutar_evaluacion_semanal_horizontes(
    df_serie: pd.DataFrame,
    min_entrenamiento: int = 16,
    horizontes: List[int] = [1, 2, 3, 4, 5]
) -> pd.DataFrame:
    """
    Ejecuta la evaluación semanal complementaria por horizonte h=1..5
    utilizando origen móvil expansivo.
    """
    validador = VentanaExpansiva(df_serie)
    y_total_serie = df_serie["citas"].values

    errores_por_h = {h: {"reales": [], "pred_ing": [], "pred_pm": []} for h in horizontes}

    for ventana in validador.divisiones_semanales(min_entrenamiento=min_entrenamiento, horizontes=horizontes):
        train_idx = ventana["train_indices"]
        y_train = y_total_serie[train_idx]

        for h, idx_test in ventana["test_por_horizonte"].items():
            y_real = y_total_serie[idx_test]
            # Predicción a horizonte h
            pred_ing = float(predecir_ingenuo_semanal(y_train, h=h)[-1])
            pred_pm = float(predecir_promedio_movil(y_train, h=h, ventana=4)[-1])

            errores_por_h[h]["reales"].append(y_real)
            errores_por_h[h]["pred_ing"].append(pred_ing)
            errores_por_h[h]["pred_pm"].append(pred_pm)

    filas_h = []
    for h in horizontes:
        reales = errores_por_h[h]["reales"]
        pred_ing = errores_por_h[h]["pred_ing"]
        pred_pm = errores_por_h[h]["pred_pm"]

        metricas_ing = calcular_todas_las_metricas(reales, pred_ing)
        metricas_pm = calcular_todas_las_metricas(reales, pred_pm, y_ingenuo=pred_ing)

        filas_h.append({
            "horizonte": f"h={h}",
            "muestras_evaluadas": len(reales),
            "MAE_ingenuo": metricas_ing["MAE"],
            "RMSE_ingenuo": metricas_ing["RMSE"],
            "MAE_promedio_movil": metricas_pm["MAE"],
            "RMSE_promedio_movil": metricas_pm["RMSE"],
            "rMAE_promedio_movil": metricas_pm["rMAE"]
        })

    return pd.DataFrame(filas_h)


def ejecutar_validacion_fase3(
    ruta_serie: Optional[Union[str, Path]] = None,
    ruta_salida_mensual: Optional[Union[str, Path]] = None,
    ruta_salida_semanal: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flujo orquestador de la Fase 3."""
    proyecto_root = Path(__file__).resolve().parent.parent.parent

    if ruta_serie is None:
        ruta_serie = proyecto_root / "data/processed/serie_semanal.csv"
    else:
        ruta_serie = Path(ruta_serie)

    if ruta_salida_mensual is None:
        ruta_salida_mensual = proyecto_root / "reports/tablas/evaluacion_mensual_origenes.csv"
    else:
        ruta_salida_mensual = Path(ruta_salida_mensual)

    if ruta_salida_semanal is None:
        ruta_salida_semanal = proyecto_root / "reports/tablas/evaluacion_semanal_horizontes.csv"
    else:
        ruta_salida_semanal = Path(ruta_salida_semanal)

    logger.info(f"Cargando serie procesada desde: {ruta_serie}")
    df_serie = pd.read_csv(ruta_serie)

    # 1. Evaluación Mensual Principal
    df_mensual = ejecutar_evaluacion_mensual_origenes(df_serie, min_meses_train=3)
    ruta_salida_mensual.parent.mkdir(parents=True, exist_ok=True)
    df_mensual.to_csv(ruta_salida_mensual, index=False, encoding="utf-8")
    logger.info(f"Resultados mensuales guardados en: {ruta_salida_mensual}")

    # 2. Evaluación Semanal Complementaria (h=1..5)
    df_semanal = ejecutar_evaluacion_semanal_horizontes(df_serie, min_entrenamiento=16, horizontes=[1, 2, 3, 4, 5])
    ruta_salida_semanal.parent.mkdir(parents=True, exist_ok=True)
    df_semanal.to_csv(ruta_salida_semanal, index=False, encoding="utf-8")
    logger.info(f"Resultados semanales guardados en: {ruta_salida_semanal}")

    return df_mensual, df_semanal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 3: Validación temporal de origen móvil y métricas")
    parser.add_argument("--serie", type=str, default=None, help="Ruta a serie_semanal.csv")
    args = parser.parse_args()
    ejecutar_validacion_fase3(ruta_serie=args.serie)
