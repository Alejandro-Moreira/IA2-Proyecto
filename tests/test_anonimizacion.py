"""
Pruebas automatizadas para la Fase 1: Anonimización (LOPDP).
Verifica que el proceso de anonimización elimine toda información personal
y que el archivo data/interim/citas_anonimas.csv contenga estricta y únicamente
las dos columnas autorizadas: 'fecha_cita' y 'estado'.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.anonimizar import (
    anonimizar_dataframe,
    ejecutar_anonimizacion,
    COLUMNAS_OBLIGATORIAS,
)


@pytest.fixture(scope="module")
def archivo_anonimizado():
    """Ejecuta el flujo de anonimización si el archivo intermedio no existe o para pruebas."""
    ruta_generada = ejecutar_anonimizacion(generar_simulado_si_no_existe=True)
    return ruta_generada


def test_anonimizar_dataframe_en_memoria():
    """
    Verifica que la función pura anonimizar_dataframe filtre correctamente
    columnas sensibles y conserve únicamente las obligatorias.
    """
    df_prueba = pd.DataFrame({
        "id_paciente": [101, 102],
        "nombre_completo": ["Juan Pérez", "María Rodríguez"],
        "cedula": ["1712345678", "1798765432"],
        "diagnostico": ["J00", "K29"],
        "fecha_cita": ["2025-09-01", "2025-09-02"],
        "estado": ["Atendida", " NO_ASISTIDA "],
        "observaciones": ["Sensible 1", "Sensible 2"]
    })

    df_resultado = anonimizar_dataframe(df_prueba)

    # Criterio estricto: exactamente 2 columnas
    assert list(df_resultado.columns) == COLUMNAS_OBLIGATORIAS
    assert len(df_resultado.columns) == 2

    # Normalización de estado
    assert df_resultado.loc[0, "estado"] == "atendida"
    assert df_resultado.loc[1, "estado"] == "no_asistida"

    # Formato de fecha
    assert df_resultado.loc[0, "fecha_cita"] == "2025-09-01"


def test_error_si_faltan_columnas_obligatorias():
    """Verifica que se lance ValueError si el archivo crudo carece de columnas obligatorias."""
    df_invalido = pd.DataFrame({
        "nombre": ["Juan"],
        "fecha_cita": ["2025-09-01"]
        # Falta 'estado'
    })

    with pytest.raises(ValueError) as excinfo:
        anonimizar_dataframe(df_invalido)
    assert "estado" in str(excinfo.value).lower()


def test_criterio_aceptacion_columnas_exactas(archivo_anonimizado):
    """
    CRITERIO DE ACEPTACIÓN PRINCIPAL:
    El archivo de salida data/interim/citas_anonimas.csv debe contener
    exactamente estas dos columnas: ['fecha_cita', 'estado']
    y ninguna columna adicional.
    """
    assert archivo_anonimizado.exists(), f"No existe el archivo {archivo_anonimizado}"

    df = pd.read_csv(archivo_anonimizado)
    columnas_obtenidas = list(df.columns)

    assert columnas_obtenidas == ["fecha_cita", "estado"], (
        f"Las columnas no coinciden con las autorizadas. "
        f"Esperadas: ['fecha_cita', 'estado'], Obtenidas: {columnas_obtenidas}"
    )
    assert len(df.columns) == 2, f"Se encontraron {len(df.columns)} columnas en lugar de 2."


def test_ausencia_total_datos_identificables_lopdp(archivo_anonimizado):
    """
    Verifica que no quede ningún dato identificable (LOPDP):
    No nombres, no cédulas, no IDs de paciente, no diagnósticos ni observaciones.
    """
    df = pd.read_csv(archivo_anonimizado)

    columnas_prohibidas = [
        "nombre", "nombre_completo", "paciente", "id_paciente",
        "cedula", "cedula_identidad", "identificacion", "telefono",
        "email", "correo", "direccion", "diagnostico", "cie10",
        "observaciones", "notas", "medico", "historia_clinica"
    ]

    for col in df.columns:
        col_lower = col.lower()
        for prohibida in columnas_prohibidas:
            assert prohibida not in col_lower, (
                f"Violación LOPDP detectada: columna sensible '{col}' presente en datos interim."
            )


def test_calidad_e_integridad_datos_anonimizados(archivo_anonimizado):
    """Verifica que no existan valores nulos y que los tipos y formatos sean consistentes."""
    df = pd.read_csv(archivo_anonimizado)

    assert len(df) > 0, "El archivo de citas anonimizadas está vacío."

    # Sin valores nulos en fecha_cita ni estado
    assert df["fecha_cita"].isna().sum() == 0, "Existen fechas de cita nulas."
    assert df["estado"].isna().sum() == 0, "Existen estados nulos."

    # Parseo de fechas válido
    fechas_convertidas = pd.to_datetime(df["fecha_cita"], errors="coerce")
    assert fechas_convertidas.isna().sum() == 0, "Existen fechas con formato inválido."

    # Estados esperados
    estados_unicos = set(df["estado"].unique())
    assert estados_unicos.issubset({"atendida", "no_asistida", "cancelada"}), (
        f"Estados desconocidos encontrados: {estados_unicos}"
    )
