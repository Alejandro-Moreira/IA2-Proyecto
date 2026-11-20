"""
Modelos de referencia y líneas base (Benchmarks) para pronóstico de demanda de citas.
Fase 4: Líneas Base y Modelos Estadísticos (Semana 8).

Implementa:
1. Modelo Ingenuo (Naive semanal y agregado):
   - Repite el último valor semanal observado multiplicado por las k semanas pronosticadas.
2. Promedio Móvil (Moving Average):
   - Calcula el promedio de las últimas k semanas históricas observadas (por defecto ventana=4).
3. Último Mes Completo:
   - Utiliza el total de demanda del mes calendario inmediatamente anterior como referencia.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


class ModeloIngenuo:
    """
    Línea base ingenua de persistencia: asume que la demanda futura mantendrá
    el último valor semanal observado en el corte temporal t.
    """

    def __init__(self, nombre: str = "Ingenuo"):
        self.nombre = nombre

    def fit(self, y_train: Union[np.ndarray, list, pd.Series]) -> "ModeloIngenuo":
        """No requiere entrenamiento con optimización; registra el último valor."""
        arr = np.asarray(y_train, dtype=float)
        if len(arr) == 0:
            raise ValueError("y_train no puede estar vacío.")
        self.ultimo_valor = float(arr[-1])
        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        """Genera predicciones para h semanas repitiendo el último valor."""
        if not hasattr(self, "ultimo_valor"):
            raise RuntimeError("Debe llamar a fit() antes de predict_semanal().")
        return np.full(shape=h, fill_value=self.ultimo_valor, dtype=float)

    def predict_mensual(self, num_semanas_mes: int) -> float:
        """Calcula la demanda mensual agregada: k * y_ultimo."""
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


class ModeloPromedioMovil:
    """
    Línea base por Promedio Móvil: calcula la media de las últimas k semanas históricas
    y proyecta ese valor promedio constante hacia las semanas futuras del mes.
    """

    def __init__(self, ventana: int = 4, nombre: str = "Promedio Móvil (k=4)"):
        self.ventana = ventana
        self.nombre = nombre
        self.promedio: Optional[float] = None

    def fit(self, y_train: Union[np.ndarray, list, pd.Series]) -> "ModeloPromedioMovil":
        arr = np.asarray(y_train, dtype=float)
        if len(arr) == 0:
            raise ValueError("y_train no puede estar vacío.")
        k = min(self.ventana, len(arr))
        self.promedio = float(np.mean(arr[-k:]))
        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        if self.promedio is None:
            raise RuntimeError("Debe llamar a fit() antes de predict_semanal().")
        return np.full(shape=h, fill_value=self.promedio, dtype=float)

    def predict_mensual(self, num_semanas_mes: int) -> float:
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


class ModeloUltimoMesCompleto:
    """
    Línea base de mes calendario previo: utiliza el total del último mes completo
    observado como referencia para predecir el siguiente mes calendario.
    Opcionalmente puede ajustar por diferencia en cantidad de semanas (4 vs 5 semanas).
    """

    def __init__(self, ajustar_por_duracion: bool = False, nombre: str = "Último Mes"):
        self.ajustar_por_duracion = ajustar_por_duracion
        self.nombre = nombre
        self.total_ultimo_mes: Optional[float] = None
        self.semanas_ultimo_mes: Optional[int] = None

    def fit_mes(
        self,
        citas_mes_previo: float,
        semanas_mes_previo: int
    ) -> "ModeloUltimoMesCompleto":
        """Registra el total y duración del mes inmediatamente anterior."""
        self.total_ultimo_mes = float(citas_mes_previo)
        self.semanas_ultimo_mes = int(semanas_mes_previo)
        return self

    def predict_mensual(self, num_semanas_mes_objetivo: int) -> float:
        if self.total_ultimo_mes is None:
            raise RuntimeError("Debe llamar a fit_mes() antes de predict_mensual().")

        if self.ajustar_por_duracion and self.semanas_ultimo_mes > 0:
            # Tasa semanal del mes previo escalada a la duración del mes objetivo
            tasa_semanal = self.total_ultimo_mes / self.semanas_ultimo_mes
            return float(tasa_semanal * num_semanas_mes_objetivo)
        else:
            # Proyección directa del total mensual previo
            return float(self.total_ultimo_mes)

    def predict_semanal(self, h: int) -> np.ndarray:
        if self.total_ultimo_mes is None:
            raise RuntimeError("Debe llamar a fit_mes() antes de predict_semanal().")
        tasa = self.total_ultimo_mes / (self.semanas_ultimo_mes or h)
        return np.full(shape=h, fill_value=tasa, dtype=float)
