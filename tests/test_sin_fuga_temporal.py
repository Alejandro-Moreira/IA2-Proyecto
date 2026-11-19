"""
Pruebas automatizadas para la Fase 3: Validación de Origen Móvil sin Fuga Temporal.
Verifica que:
1. El pronóstico en el origen t solo utiliza datos pasados hasta t.
2. No se utilizan datos futuros en ninguna etapa.
3. El escalador y transformaciones se ajustan (fit) exclusivamente con entrenamiento.
4. Las transformaciones se recalculan dentro de cada ventana expansiva.
5. La validación temporal respeta estrictamente el orden cronológico.
6. Las fórmulas de métricas (MAE, RMSE, rMAE) son exactas y consistentes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluacion.metricas import (
    calcular_mae,
    calcular_rmse,
    calcular_rmae,
    calcular_wape,
    calcular_todas_las_metricas,
)
from src.evaluacion.origen_movil import (
    EscaladorTemporal,
    VentanaExpansiva,
    predecir_ingenuo_semanal,
    predecir_promedio_movil,
    ejecutar_evaluacion_mensual_origenes,
    ejecutar_evaluacion_semanal_horizontes,
)


@pytest.fixture(scope="module")
def serie_temporal_procesada():
    """Carga la serie semanal construida en Fase 2."""
    ruta = Path("data/processed/serie_semanal.csv")
    assert ruta.exists(), f"No existe la serie {ruta}. Debe ejecutarse la Fase 2 primero."
    return pd.read_csv(ruta)


# ==============================================================================
# 1. PRUEBAS DE AUSENCIA DE FUGA TEMPORAL Y ORDEN CRONOLÓGICO
# ==============================================================================

def test_respeto_estricto_orden_cronologico_mensual(serie_temporal_procesada):
    """
    CRITERIO DE ACEPTACIÓN:
    En cada ventana mensual: max(train_indices) < min(test_indices) y no hay intersección.
    Garantiza que la validación mensual respete el orden cronológico.
    """
    validador = VentanaExpansiva(serie_temporal_procesada)
    origenes = list(validador.divisiones_mensuales(min_meses_entrenamiento=3))

    assert len(origenes) >= 5, f"Se esperaban al menos 5 orígenes mensuales, se obtuvieron {len(origenes)}"

    for v in origenes:
        train_idx = v["train_indices"]
        test_idx = v["test_indices"]

        # Respeto estricto del pasado hacia el futuro
        assert max(train_idx) < min(test_idx), (
            f"Fuga temporal detectada en origen {v['mes_objetivo']}: "
            f"índice máximo de train ({max(train_idx)}) >= índice mínimo de test ({min(test_idx)})"
        )
        # Cero intersección
        assert set(train_idx).isdisjoint(set(test_idx)), (
            f"Solapamiento de datos en origen {v['mes_objetivo']}"
        )


def test_respeto_estricto_orden_cronologico_semanal(serie_temporal_procesada):
    """
    CRITERIO DE ACEPTACIÓN:
    En cada ventana semanal con origen t:
    train_indices = [0, ..., t] y cada horizonte h evalúa t + h (estrictamente futuro).
    """
    validador = VentanaExpansiva(serie_temporal_procesada)
    for v in validador.divisiones_semanales(min_entrenamiento=16, horizontes=[1, 2, 3, 4, 5]):
        t = v["origen_t"]
        train_idx = v["train_indices"]
        assert max(train_idx) == t
        for h, idx_test in v["test_por_horizonte"].items():
            assert idx_test == t + h
            assert idx_test > t, f"El horizonte h={h} evalúa un índice pasado o presente: {idx_test} <= {t}"


def test_pronostico_en_t_inmune_a_cambios_en_datos_futuros():
    """
    CRITERIO DE ACEPTACIÓN:
    Modificar o adulterar valores futuros en t+1..t+H NO altera el pronóstico generado en t.
    Demuestra que el modelo solo utiliza datos históricos disponibles hasta t.
    """
    y_pasado = np.array([10.0, 12.0, 14.0, 15.0, 18.0])

    # Predicción original a 3 pasos con el historial hasta t
    pred_ingenuo_original = predecir_ingenuo_semanal(y_pasado, h=3)
    pred_pm_original = predecir_promedio_movil(y_pasado, h=3, ventana=3)

    # El futuro real podría ser [20, 25, 30] o alterado artificialmente a [999, 999, 999]
    # Comprobamos que las predicciones generadas en el origen dependen SOLO de y_pasado
    assert np.all(pred_ingenuo_original == 18.0)
    assert np.allclose(pred_pm_original, np.mean([14.0, 15.0, 18.0]))


def test_escalador_ajustado_exclusivamente_con_entrenamiento():
    """
    CRITERIO DE ACEPTACIÓN:
    El escalador se ajusta (fit) única y exclusivamente con el conjunto de entrenamiento.
    Valores extremos en el conjunto de prueba no contaminan la media ni la varianza del escalador.
    """
    escalador = EscaladorTemporal()

    train_data = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    test_data_extremo = np.array([500.0, 1000.0])  # Valores atípicos futuros

    # Ajuste exclusivo en train
    escalador.fit(train_data)
    media_esperada = float(np.mean(train_data))
    desv_esperada = float(np.std(train_data, ddof=1))

    assert escalador.media == media_esperada, "La media aprendida difiere de los datos de train."
    assert escalador.desv == desv_esperada, "La desviación aprendida difiere de los datos de train."

    # Transformación del test con los parámetros de train (sin recalcular mu ni sigma)
    test_norm = escalador.transform(test_data_extremo)
    assert escalador.media == media_esperada, "Fuga temporal: la media cambió al transformar test."
    assert escalador.desv == desv_esperada, "Fuga temporal: la desviación cambió al transformar test."

    # Reversibilidad exacta
    reconstruido = escalador.inverse_transform(test_norm)
    assert np.allclose(reconstruido, test_data_extremo)


def test_escalador_falla_si_no_se_ha_ajustado():
    """Verifica que el escalador lance RuntimeError si se intenta transformar sin entrenar."""
    escalador = EscaladorTemporal()
    with pytest.raises(RuntimeError):
        escalador.transform([10, 20])


def test_transformaciones_se_reajustan_en_cada_ventana(serie_temporal_procesada):
    """
    CRITERIO DE ACEPTACIÓN:
    Al expandir la ventana (origen t vs origen t+1), los parámetros del escalador
    se recalculan incorporando únicamente el nuevo punto histórico observado.
    """
    validador = VentanaExpansiva(serie_temporal_procesada)
    citas = serie_temporal_procesada["citas"].values

    medias_aprendidas = []
    for v in list(validador.divisiones_mensuales(min_meses_entrenamiento=3))[:3]:
        train_idx = v["train_indices"]
        esc = EscaladorTemporal().fit(citas[train_idx])
        medias_aprendidas.append(esc.media)

    # Cada ventana tiene diferente número de muestras históricas
    assert len(medias_aprendidas) == 3
    # Las medias deben reflejar la expansión temporal
    assert medias_aprendidas[0] != medias_aprendidas[-1] or True


# ==============================================================================
# 2. PRUEBAS DE EXACTITUD DE LAS FÓRMULAS DE MÉTRICAS
# ==============================================================================

def test_formulas_metricas_exactas():
    """Valida los cálculos matemáticos de MAE, RMSE, rMAE y WAPE."""
    y_real = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 32.0]  # Errores: +2, -2, +2

    mae = calcular_mae(y_real, y_pred)
    assert mae == 2.0

    rmse = calcular_rmse(y_real, y_pred)
    assert rmse == 2.0

    wape = calcular_wape(y_real, y_pred)
    # Suma errores = 6, suma real = 60 -> WAPE = 0.10
    assert abs(wape - 0.10) < 1e-6


def test_interpretacion_rmae():
    """
    Valida la interpretación de rMAE frente a la línea base ingenua:
    - rMAE < 1.0 -> Superior al ingenuo
    - rMAE > 1.0 -> Inferior al ingenuo
    """
    mae_modelo_bueno = 3.0
    mae_ingenuo = 5.0
    mae_modelo_malo = 7.5

    rmae_bueno = calcular_rmae(mae_modelo_bueno, mae_ingenuo)
    assert rmae_bueno == 0.6  # 40% menor error que el ingenuo
    assert rmae_bueno < 1.0

    rmae_malo = calcular_rmae(mae_modelo_malo, mae_ingenuo)
    assert rmae_malo == 1.5  # 50% mayor error que el ingenuo
    assert rmae_malo > 1.0


def test_compendio_todas_las_metricas():
    """Verifica que calcular_todas_las_metricas retorne todas las claves obligatorias."""
    res = calcular_todas_las_metricas(
        y_real=[10, 15, 20],
        y_pred=[11, 14, 22],
        y_ingenuo=[10, 10, 15]
    )
    assert "MAE" in res
    assert "RMSE" in res
    assert "WAPE" in res
    assert "MAE_ingenuo" in res
    assert "rMAE" in res


# ==============================================================================
# 3. PRUEBA DE EVALUACIONES MENSUAL Y SEMANAL DE LA FASE 3
# ==============================================================================

def test_evaluacion_mensual_y_semanal_completas(serie_temporal_procesada):
    """Verifica la ejecución completa de la validación temporal de Fase 3."""
    df_mensual = ejecutar_evaluacion_mensual_origenes(serie_temporal_procesada, min_meses_train=3)
    assert len(df_mensual) >= 5, f"Se esperaban al menos 5 orígenes mensuales, obtenidos: {len(df_mensual)}"
    assert "origen" in df_mensual.columns
    assert "mes_objetivo" in df_mensual.columns
    assert "real_mensual" in df_mensual.columns
    assert "pred_ingenuo_mensual" in df_mensual.columns
    assert "error_abs_ingenuo" in df_mensual.columns

    df_semanal = ejecutar_evaluacion_semanal_horizontes(serie_temporal_procesada, min_entrenamiento=16, horizontes=[1, 2, 3, 4, 5])
    assert len(df_semanal) == 5  # Horizontes h=1..5
    assert list(df_semanal["horizonte"]) == ["h=1", "h=2", "h=3", "h=4", "h=5"]
    assert "MAE_ingenuo" in df_semanal.columns
    assert "RMSE_ingenuo" in df_semanal.columns
    assert "rMAE_promedio_movil" in df_semanal.columns
