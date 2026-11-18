"""
Pruebas automatizadas para la Fase 2: Construcción de la Serie Semanal.
Verifica columnas obligatorias, regla del jueves, cálculo de estadísticas,
semanas operativas/cerradas y evaluación del Gate Metodológico de 30 semanas.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.construir_serie import (
    construir_serie_semanal,
    generar_estadisticas_dataset,
    evaluar_gate_metodologico,
    ejecutar_construccion_serie,
    asignar_mes_regla_jueves,
    COLUMNAS_OBLIGATORIAS,
)


@pytest.fixture(scope="module")
def serie_procesada():
    """Ejecuta el pipeline de Fase 2 para generar los artefactos necesarios."""
    df_serie, resultado_gate = ejecutar_construccion_serie(tipo_demanda="principal")
    return df_serie, resultado_gate


def test_columnas_obligatorias_serie_semanal(serie_procesada):
    """
    CRITERIO DE ACEPTACIÓN:
    La serie semanal debe contener exactamente las columnas obligatorias:
    semana_inicio, citas, operativa, mes_objetivo.
    """
    df_serie, _ = serie_procesada
    ruta_serie = Path("data/processed/serie_semanal.csv")

    assert ruta_serie.exists(), f"El archivo procesado no existe: {ruta_serie}"
    df_csv = pd.read_csv(ruta_serie)

    assert list(df_csv.columns) == COLUMNAS_OBLIGATORIAS
    assert len(df_csv) > 0, "La serie semanal generada no contiene filas."

    # Tipos válidos
    assert pd.to_datetime(df_csv["semana_inicio"], errors="coerce").isna().sum() == 0
    assert df_csv["citas"].dtype in [int, "int64"]
    assert df_csv["operativa"].dtype in [bool, "bool"]
    assert df_csv["mes_objetivo"].dtype in [int, "int64"]


def test_regla_del_jueves(serie_procesada):
    """
    Verifica que la asignación de cada semana a un mes objetivo
    siga estrictamente la regla del jueves: (semana_inicio + 3 días).month.
    """
    df_serie, _ = serie_procesada
    for _, fila in df_serie.iterrows():
        sem_inicio = pd.to_datetime(fila["semana_inicio"])
        mes_esperado = (sem_inicio + pd.Timedelta(days=3)).month
        assert int(fila["mes_objetivo"]) == mes_esperado, (
            f"Fallo en regla del jueves para semana {fila['semana_inicio']}: "
            f"obtenido={fila['mes_objetivo']}, esperado={mes_esperado}"
        )


def test_tabla_estadisticas_dataset():
    """
    CRITERIO DE ACEPTACIÓN:
    El archivo reports/tablas/dataset.csv debe existir y contener:
    - semanas_totales, semanas_validas, semanas_excluidas, total_citas,
      media, desviacion_estandar, minimo, maximo.
    """
    ruta_stats = Path("reports/tablas/dataset.csv")
    assert ruta_stats.exists(), f"La tabla de estadísticas no existe: {ruta_stats}"

    df_stats = pd.read_csv(ruta_stats)
    metricas = set(df_stats["metrica"].values)

    metricas_requeridas = {
        "semanas_totales",
        "semanas_validas",
        "semanas_excluidas",
        "total_citas",
        "media",
        "desviacion_estandar",
        "minimo",
        "maximo"
    }

    assert metricas_requeridas.issubset(metricas), (
        f"Faltan métricas obligatorias en dataset.csv. Faltantes: {metricas_requeridas - metricas}"
    )

    valores = dict(zip(df_stats["metrica"], df_stats["valor"]))

    # Coherencia matemática
    assert valores["semanas_validas"] + valores["semanas_excluidas"] == valores["semanas_totales"]
    assert valores["minimo"] <= valores["media"] <= valores["maximo"]
    assert valores["total_citas"] >= 0


def test_evaluacion_gate_metodologico():
    """Verifica que el Gate Metodológico de 30 semanas funcione y active el Plan B si procede."""
    # Caso >= 30 semanas: Aprobado Plan A
    res_aprobado = evaluar_gate_metodologico(semanas_validas=35, umbral=30)
    assert res_aprobado["gate_superado"] is True
    assert "APROBADO" in res_aprobado["dictamen"]

    # Caso < 30 semanas: Activación Plan B
    res_plan_b = evaluar_gate_metodologico(semanas_validas=22, umbral=30)
    assert res_plan_b["gate_superado"] is False
    assert "PLAN B ACTIVADO" in res_plan_b["dictamen"]


def test_analisis_sensibilidad_definicion_demanda():
    """Verifica que la demanda principal incluya citas no asistidas frente al análisis de sensibilidad."""
    df_prueba = pd.DataFrame({
        "fecha_cita": ["2025-09-01", "2025-09-02", "2025-09-03"],
        "estado": ["atendida", "no_asistida", "cancelada"]
    })

    serie_principal = construir_serie_semanal(df_prueba, tipo_demanda="principal")
    serie_sensibilidad = construir_serie_semanal(df_prueba, tipo_demanda="sensibilidad")

    # Demanda principal = atendida (1) + no asistida (1) = 2
    assert serie_principal.loc[0, "citas"] == 2

    # Sensibilidad = solo atendida = 1
    assert serie_sensibilidad.loc[0, "citas"] == 1


def test_semanas_cierre_no_operativas():
    """Verifica que las semanas marcadas en semanas_cierre tengan operativa=False."""
    df_prueba = pd.DataFrame({
        "fecha_cita": ["2025-09-01", "2025-12-29"],
        "estado": ["atendida", "atendida"]
    })

    serie = construir_serie_semanal(
        df_prueba,
        semanas_cierre=["2025-12-29"]
    )

    fila_cierre = serie[serie["semana_inicio"] == "2025-12-29"]
    assert len(fila_cierre) == 1
    assert fila_cierre.iloc[0]["operativa"] == False


def test_figura_serie_semanal_generada():
    """Verifica que el gráfico de la serie semanal exista y contenga datos."""
    ruta_figura = Path("reports/figuras/serie_semanal.png")
    assert ruta_figura.exists(), f"No existe la figura {ruta_figura}"
    assert ruta_figura.stat().st_size > 0, "La figura generada está vacía (0 bytes)."
