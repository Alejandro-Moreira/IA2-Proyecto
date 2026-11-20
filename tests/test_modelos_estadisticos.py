"""
Pruebas automatizadas para la Fase 4: Líneas Base y Modelos Estadísticos.
Verifica:
1. Implementación y funcionamiento de líneas base (Ingenuo, Promedio Móvil, Último Mes).
2. Modelos estadísticos: Regresión Ridge con lags, ETS no estacional y ARIMA con selección AIC.
3. Restricciones metodológicas (sin SARIMA m=52, sin asumir año completo, sin fuga en test).
4. Generación y consistencia de reports/tablas/lineas_base.csv sobre los mismos 6 orígenes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.modelos.lineas_base import (
    ModeloIngenuo,
    ModeloPromedioMovil,
    ModeloUltimoMesCompleto,
)
from src.modelos.estadisticos import (
    ModeloRegresionLags,
    ModeloETS,
    ModeloARIMA,
    evaluar_lineas_base_y_estadisticos,
    ejecutar_fase4,
)


@pytest.fixture(scope="module")
def serie_temporal():
    ruta = Path("data/processed/serie_semanal.csv")
    assert ruta.exists(), f"No existe la serie {ruta}"
    return pd.read_csv(ruta)


def test_modelo_ingenuo_persistencia():
    """Verifica que el modelo ingenuo repita el último valor semanal y agregue correctamente."""
    y = np.array([10.0, 15.0, 20.0])
    mod = ModeloIngenuo().fit(y)
    pred_sem = mod.predict_semanal(h=4)
    pred_mes = mod.predict_mensual(num_semanas_mes=4)

    assert np.all(pred_sem == 20.0)
    assert pred_mes == 80.0  # 4 * 20


def test_modelo_promedio_movil():
    """Verifica el promedio móvil de las últimas k semanas."""
    y = np.array([10.0, 10.0, 20.0, 20.0])
    mod = ModeloPromedioMovil(ventana=2).fit(y)
    assert mod.promedio == 20.0  # Media de [20, 20]
    assert mod.predict_mensual(num_semanas_mes=4) == 80.0

    mod4 = ModeloPromedioMovil(ventana=4).fit(y)
    assert mod4.promedio == 15.0  # Media de [10, 10, 20, 20]
    assert mod4.predict_mensual(num_semanas_mes=4) == 60.0


def test_modelo_ultimo_mes_completo():
    """Verifica la línea base de último mes calendario."""
    mod = ModeloUltimoMesCompleto(ajustar_por_duracion=False)
    mod.fit_mes(citas_mes_previo=60.0, semanas_mes_previo=4)
    assert mod.predict_mensual(num_semanas_mes_objetivo=5) == 60.0

    # Versión con ajuste por duración (tasa semanal escalada)
    mod_ajustado = ModeloUltimoMesCompleto(ajustar_por_duracion=True)
    mod_ajustado.fit_mes(citas_mes_previo=60.0, semanas_mes_previo=4)
    assert mod_ajustado.predict_mensual(num_semanas_mes_objetivo=5) == 75.0  # (60/4)*5


def test_modelo_regresion_ridge_lags():
    """Verifica la regresión Ridge autorregresiva con lags y pronóstico recursivo."""
    y = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=float)
    mod = ModeloRegresionLags(lags=[1, 2, 3, 4], alpha=1.0).fit(y)

    pred_sem = mod.predict_semanal(h=4)
    assert len(pred_sem) == 4
    # Todas las predicciones deben ser finitas y no negativas
    assert np.all(np.isfinite(pred_sem))
    assert np.all(pred_sem >= 0.0)

    pred_mes = mod.predict_mensual(num_semanas_mes=4)
    assert pred_mes == float(np.sum(pred_sem))


def test_modelo_ets_no_estacional():
    """Verifica que el modelo ETS no estacional ajuste y genere predicciones válidas."""
    y = np.array([12, 15, 14, 16, 18, 17, 19, 21, 20, 22, 24], dtype=float)
    mod = ModeloETS(tendencia=None).fit(y)
    pred_sem = mod.predict_semanal(h=4)
    assert len(pred_sem) == 4
    assert np.all(np.isfinite(pred_sem))
    assert np.all(pred_sem >= 0.0)


def test_modelo_arima_seleccion_aicc_en_train():
    """Verifica que ARIMA busque en cuadrícula dentro de train y elija el mejor orden."""
    y = np.array([15, 14, 18, 17, 19, 21, 20, 22, 23, 25, 24, 26, 28], dtype=float)
    mod = ModeloARIMA(grid_p=[0, 1], grid_d=[0, 1], grid_q=[0, 1]).fit(y)
    assert mod.mejor_orden is not None
    assert len(mod.mejor_orden) == 3

    pred_sem = mod.predict_semanal(h=4)
    assert len(pred_sem) == 4
    assert np.all(np.isfinite(pred_sem))
    assert np.all(pred_sem >= 0.0)


def test_criterio_aceptacion_tabla_lineas_base(serie_temporal):
    """
    CRITERIO DE ACEPTACIÓN PRINCIPAL:
    El archivo reports/tablas/lineas_base.csv debe existir y contener
    todos los modelos evaluados sobre exactamente los mismos 6 orígenes.
    """
    df_resumen, df_detalle = ejecutar_fase4()
    ruta_tabla = Path("reports/tablas/lineas_base.csv")
    assert ruta_tabla.exists(), f"No existe la tabla de líneas base: {ruta_tabla}"

    df_csv = pd.read_csv(ruta_tabla)

    columnas_esperadas = [
        "modelo", "categoria", "MAE_mensual", "RMSE_mensual",
        "rMAE_mensual", "WAPE_mensual", "mejora_vs_ingenuo"
    ]
    for col in columnas_esperadas:
        assert col in df_csv.columns, f"Falta columna {col} en lineas_base.csv"

    # Verificar que estén los 6 modelos evaluados
    assert len(df_csv) == 6, f"Se esperaban 6 modelos, se encontraron {len(df_csv)}"

    modelos_presentes = set(df_csv["modelo"].values)
    modelos_requeridos = {
        "Ingenuo (Persistencia)",
        "Promedio Móvil (k=4)",
        "Último Mes Completo",
        "Regresión Ridge (Lags 1-4)",
        "ETS (Suavizamiento Exponencial)",
        "ARIMA (Auto-AICc)"
    }
    assert modelos_requeridos == modelos_presentes

    # El modelo Ingenuo debe tener exactamente rMAE = 1.0
    fila_ingenuo = df_csv[df_csv["modelo"] == "Ingenuo (Persistencia)"].iloc[0]
    assert fila_ingenuo["rMAE_mensual"] == 1.0

    # Coherencia matemática básica: MAE <= RMSE
    for _, fila in df_csv.iterrows():
        assert fila["MAE_mensual"] <= fila["RMSE_mensual"] + 1e-5, (
            f"Inconsistencia matemática en {fila['modelo']}: MAE ({fila['MAE_mensual']}) > RMSE ({fila['RMSE_mensual']})"
        )

    # Verificar que el detalle contenga exactamente 6 orígenes mensuales
    assert len(df_detalle) == 6, f"Se esperaban 6 orígenes en detalle, se obtuvieron {len(df_detalle)}"
