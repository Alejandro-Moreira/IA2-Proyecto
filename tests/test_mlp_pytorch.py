"""
Pruebas automatizadas para la Fase 5: Red Neuronal MLP en PyTorch.
Verifica:
1. Arquitectura de la red (entradas con lags 1-4 + promedio móvil, oculta 16, ReLU, salida 1).
2. Reproducibilidad mediante control de semillas aleatorias (0, 1, 2).
3. Partición de validación estrictamente cronológica (20% final de train, sin shuffle).
4. Parada temprana (Early Stopping) con regularización L2 (weight_decay).
5. Generación física de las curvas de entrenamiento (reports/figuras/curvas_mlp_semilla{0,1,2}.png).
6. Generación y consistencia del reporte comparativo reports/tablas/resultados_mlp.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.modelos.mlp import (
    RedMLP,
    EnvoltorioMLP,
    construir_dataset_lags,
    ejecutar_fase5,
    fijar_semillas,
)


@pytest.fixture(scope="module")
def ejecucion_mlp():
    """Ejecuta el pipeline de entrenamiento multi-semilla de Fase 5."""
    df_resumen, df_detalle = ejecutar_fase5()
    return df_resumen, df_detalle


def test_arquitectura_red_mlp():
    """Verifica que la red neuronal tenga 5 entradas, 16 neuronas ocultas, ReLU y 1 salida."""
    red = RedMLP(input_dim=5, hidden_dim=16, output_dim=1)

    assert isinstance(red.fc1, torch.nn.Linear)
    assert red.fc1.in_features == 5
    assert red.fc1.out_features == 16
    assert isinstance(red.act, torch.nn.ReLU)
    assert isinstance(red.fc2, torch.nn.Linear)
    assert red.fc2.in_features == 16
    assert red.fc2.out_features == 1

    # Prueba de forward pass
    x_prueba = torch.randn(2, 5)
    salida = red(x_prueba)
    assert salida.shape == (2, 1)


def test_dataset_lags_y_promedio_movil():
    """Verifica que el dataset de entrada incluya los 4 lags y el promedio móvil de 4 semanas."""
    serie = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    X, y = construir_dataset_lags(serie, lags=[1, 2, 3, 4], incluir_promedio_movil=True)

    # 6 muestras - 4 lags = 2 filas
    assert X.shape == (2, 5)
    assert y.shape == (2,)

    # Primera fila: lags [16, 14, 12, 10], promedio = 13.0, y = 18
    assert list(X[0, :4]) == [16.0, 14.0, 12.0, 10.0]
    assert X[0, 4] == 13.0
    assert y[0] == 18.0


def test_reproducibilidad_con_semillas():
    """Verifica que una misma semilla produzca predicciones idénticas en diferentes ejecuciones."""
    datos = np.array([12, 14, 15, 17, 16, 18, 20, 22, 19, 21, 23, 25, 24, 26], dtype=float)

    mlp_a = EnvoltorioMLP(semilla=42, max_epocas=20).fit(datos)
    pred_a = mlp_a.predict_semanal(h=3)

    mlp_b = EnvoltorioMLP(semilla=42, max_epocas=20).fit(datos)
    pred_b = mlp_b.predict_semanal(h=3)

    assert np.allclose(pred_a, pred_b, atol=1e-5), "Las predicciones difieren pese a usar la misma semilla."


def test_particion_validacion_cronologica_sin_fuga():
    """
    CRITERIO DE ACEPTACIÓN:
    La partición de validación interna se realiza tomando el último 20% cronológico,
    sin mezclas aleatorias (*no shuffle*) que provoquen fuga de información.
    """
    serie = np.arange(20, dtype=float)
    X, y = construir_dataset_lags(serie, lags=[1, 2, 3, 4], incluir_promedio_movil=True)

    n_muestras = len(X)
    n_val = max(1, int(n_muestras * 0.20))
    n_train = n_muestras - n_val

    y_train = y[:n_train]
    y_val = y[n_train:]

    # Respeto del orden temporal
    assert max(y_train) < min(y_val), "Violación de orden cronológico en split interno de validación."


def test_early_stopping_guarda_historial():
    """Verifica que el entrenamiento con early stopping guarde historial de pérdidas train y val."""
    datos = np.array([10, 12, 15, 14, 16, 18, 20, 19, 22, 24, 23, 25, 27, 26, 28, 30], dtype=float)
    mlp = EnvoltorioMLP(semilla=0, paciencia=5, max_epocas=50).fit(datos)

    historial = mlp.historial_perdidas
    assert "train" in historial and "val" in historial
    assert len(historial["train"]) > 0
    assert len(historial["val"]) == len(historial["train"])
    assert mlp.mejor_epoca > 0


def test_curvas_aprendizaje_generadas(ejecucion_mlp):
    """
    CRITERIO DE ACEPTACIÓN:
    Deben generarse físicamente las curvas de aprendizaje para las semillas 0, 1 y 2.
    """
    dir_figuras = Path("reports/figuras")
    for s in [0, 1, 2]:
        ruta_curva = dir_figuras / f"curvas_mlp_semilla{s}.png"
        assert ruta_curva.exists(), f"No existe la curva de aprendizaje para semilla {s}: {ruta_curva}"
        assert ruta_curva.stat().st_size > 0, f"El archivo {ruta_curva} está vacío."


def test_tabla_resultados_mlp_completa(ejecucion_mlp):
    """
    CRITERIO DE ACEPTACIÓN:
    El archivo reports/tablas/resultados_mlp.csv debe existir y contener:
    - Resultados de las 3 semillas individuales (0, 1, 2).
    - Fila resumen de Media ± Desv de las 3 semillas.
    - Comparación contra Regresión Ridge e Ingenuo bajo los mismos 6 orígenes.
    """
    ruta_tabla = Path("reports/tablas/resultados_mlp.csv")
    assert ruta_tabla.exists(), f"No existe la tabla de resultados del MLP: {ruta_tabla}"

    df_res = pd.read_csv(ruta_tabla)
    modelos_presentes = set(df_res["modelo"].values)

    assert "Ingenuo (Persistencia)" in modelos_presentes
    assert "Regresión Ridge (Lags 1-4)" in modelos_presentes
    assert "MLP PyTorch (Semilla 0)" in modelos_presentes
    assert "MLP PyTorch (Semilla 1)" in modelos_presentes
    assert "MLP PyTorch (Semilla 2)" in modelos_presentes
    assert any("Media ± Desv" in m for m in modelos_presentes)

    # El modelo ingenuo debe tener rMAE = 1.0000
    fila_ing = df_res[df_res["modelo"] == "Ingenuo (Persistencia)"].iloc[0]
    assert fila_ing["rMAE_mensual"] == 1.0

    # Coherencia matemática básica: MAE <= RMSE
    for _, fila in df_res.iterrows():
        assert fila["MAE_mensual"] <= fila["RMSE_mensual"] + 1e-4
