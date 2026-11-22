"""
Pruebas automatizadas para la Fase 6: Reporte Reproducible.
Verifica:
1. Regeneración completa de resultados con un único comando.
2. Generación de las cuatro figuras principales obligatorias.
3. Conservación de las curvas individuales de las 3 semillas.
4. Generación y coherencia de reports/tablas/tabla_comparativa_global.csv.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.evaluacion.generar_reporte import ejecutar_reporte_completo


@pytest.fixture(scope="module")
def reporte_ejecutado():
    """Ejecuta el pipeline completo de Fase 6."""
    return ejecutar_reporte_completo()


def test_ejecucion_generador_reporte(reporte_ejecutado):
    """CRITERIO DE ACEPTACIÓN: Un único comando ejecuta y retorna todas las tablas y figuras."""
    assert "tabla_global" in reporte_ejecutado
    assert "fig1" in reporte_ejecutado
    assert "fig2" in reporte_ejecutado
    assert "fig3" in reporte_ejecutado
    assert "fig4" in reporte_ejecutado


def test_existencia_cuatro_figuras_principales(reporte_ejecutado):
    """
    CRITERIO DE ACEPTACIÓN:
    Deben existir físicamente las 4 figuras principales requeridas:
    1. serie_temporal_completa.png
    2. pronostico_mensual_vs_real.png
    3. curvas_entrenamiento_mlp.png
    4. error_por_origen.png
    """
    figuras_esperadas = [
        "reports/figuras/serie_temporal_completa.png",
        "reports/figuras/pronostico_mensual_vs_real.png",
        "reports/figuras/curvas_entrenamiento_mlp.png",
        "reports/figuras/error_por_origen.png",
        "reports/figuras/curvas_mlp_semilla0.png",
        "reports/figuras/curvas_mlp_semilla1.png",
        "reports/figuras/curvas_mlp_semilla2.png"
    ]

    for ruta_str in figuras_esperadas:
        p = Path(ruta_str)
        assert p.exists(), f"No existe la figura requerida: {p}"
        assert p.stat().st_size > 0, f"La figura {p} está vacía."


def test_tabla_comparativa_global_contenido(reporte_ejecutado):
    """
    CRITERIO DE ACEPTACIÓN:
    El archivo reports/tablas/tabla_comparativa_global.csv debe contener todos
    los modelos evaluados, categorizados y ordenados por desempeño.
    """
    ruta_tabla = Path("reports/tablas/tabla_comparativa_global.csv")
    assert ruta_tabla.exists(), f"No existe {ruta_tabla}"

    df = pd.read_csv(ruta_tabla)

    columnas_obligatorias = [
        "modelo", "categoria", "MAE_mensual", "RMSE_mensual",
        "rMAE_mensual", "WAPE_mensual", "mejora_vs_ingenuo"
    ]
    for col in columnas_obligatorias:
        assert col in df.columns, f"Falta columna obligatoria {col} en tabla global."

    # Categorías representadas
    categorias = set(df["categoria"].dropna().unique())
    assert "Línea Base" in categorias
    assert "Modelo Estadístico" in categorias
    assert "Red Neuronal" in categorias

    modelos_presentes = set(df["modelo"].values)
    modelos_requeridos = {
        "Ingenuo (Persistencia)",
        "Promedio Móvil (k=4)",
        "Regresión Ridge (Lags 1-4)",
        "ETS (Suavizamiento Exponencial)",
        "ARIMA (Auto-AICc)",
        "MLP PyTorch (Semilla 0)",
        "MLP PyTorch (Semilla 1)",
        "MLP PyTorch (Semilla 2)",
    }
    assert modelos_requeridos.issubset(modelos_presentes)

    # El modelo ingenuo debe tener rMAE = 1.0000
    fila_ing = df[df["modelo"] == "Ingenuo (Persistencia)"].iloc[0]
    assert fila_ing["rMAE_mensual"] == 1.0


def test_coherencia_metrica_orden_descendente(reporte_ejecutado):
    """Verifica que la tabla comparativa global esté ordenada ascendentemente por error (menor MAE primero)."""
    ruta_tabla = Path("reports/tablas/tabla_comparativa_global.csv")
    df = pd.read_csv(ruta_tabla)

    maes = df["MAE_mensual"].tolist()
    assert maes == sorted(maes), "La tabla global no está ordenada ascendentemente por MAE_mensual."

    for _, fila in df.iterrows():
        assert fila["MAE_mensual"] <= fila["RMSE_mensual"] + 1e-4, (
            f"Inconsistencia matemática en {fila['modelo']}: MAE > RMSE"
        )
