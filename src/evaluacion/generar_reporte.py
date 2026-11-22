"""
Generador automatizado de reportes, tablas comparativas y figuras reproducibles.
Fase 6: Reporte Reproducible (Semana 10).

Permite regenerar todo el conjunto de resultados con un único comando:
    python src/evaluacion/generar_reporte.py

Genera:
1. Tabla comparativa global con todos los modelos:
   - reports/tablas/tabla_comparativa_global.csv
2. Cuatro figuras principales requeridas:
   - reports/figuras/serie_temporal_completa.png
   - reports/figuras/pronostico_mensual_vs_real.png
   - reports/figuras/curvas_entrenamiento_mlp.png
   - reports/figuras/error_por_origen.png
   (conservando además las curvas individuales curvas_mlp_semilla{0,1,2}.png).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Asegurar importación de src
PROYECTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROYECTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROYECTO_ROOT))

from src.evaluacion.metricas import (
    calcular_mae,
    calcular_rmse,
    calcular_rmae,
    calcular_wape,
)
from src.modelos.estadisticos import evaluar_lineas_base_y_estadisticos
from src.modelos.mlp import evaluar_mlp_tres_semillas, graficar_curvas_aprendizaje

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.reporte")


# ==============================================================================
# 1. GENERACIÓN DE FIGURAS PRINCIPALES
# ==============================================================================

def generar_figura_1_serie_completa(
    df_serie: pd.DataFrame,
    ruta_salida: Union[str, Path]
) -> Path:
    """
    Figura 1: Serie temporal semanal completa con semanas cerradas/no operativas
    explícitamente marcadas, líneas divisorias mensuales y media histórica.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)

    x = range(len(df_serie))
    y = df_serie["citas"].values
    operativas = df_serie["operativa"].values

    # Graficar línea de demanda semanal
    ax.plot(x, y, color="#1f77b4", marker="o", markersize=5, linewidth=2, label="Citas semanales válidas")

    # Resaltar semanas cerradas (no operativas)
    no_op_idx = [i for i, op in enumerate(operativas) if not op]
    if no_op_idx:
        ax.scatter(
            no_op_idx,
            [y[i] for i in no_op_idx],
            color="#d62728",
            s=120,
            zorder=5,
            label="Semana cerrada / Receso (Operativa=False)"
        )
        for idx in no_op_idx:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#d62728", alpha=0.15)

    # Línea de media histórica
    media_val = float(df_serie[df_serie["operativa"] == True]["citas"].mean())
    ax.axhline(media_val, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Media histórica ({media_val:.1f} citas/sem)")

    # Etiquetas de fechas
    paso = max(1, len(df_serie) // 14)
    ticks_idx = list(range(0, len(df_serie), paso))
    ax.set_xticks(ticks_idx)
    ax.set_xticklabels([df_serie.loc[i, "semana_inicio"] for i in ticks_idx], rotation=40, ha="right", fontsize=9)

    ax.set_title("Figura 1: Serie Temporal Semanal de Demanda de Citas Médicas (Sep 2025 - Jun 2026)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Semana de Inicio (Lunes ISO)", fontsize=10)
    ax.set_ylabel("Citas Semanales", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida


def generar_figura_2_pronostico_vs_real(
    df_detalle: pd.DataFrame,
    ruta_salida: Union[str, Path]
) -> Path:
    """
    Figura 2: Pronóstico mensual frente a valores reales a través de los 6 orígenes
    temporales para las familias clave: Real vs Ingenuo vs Promedio Móvil vs Ridge vs MLP.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)

    meses = df_detalle["mes_objetivo"].tolist()
    x = np.arange(len(meses))
    ancho = 0.16

    reales = df_detalle["real_mensual"].values
    ingenuo = df_detalle["pred_Ingenuo"].values
    pm4 = df_detalle["pred_Promedio Móvil (k=4)"].values if "pred_Promedio Móvil (k=4)" in df_detalle else df_detalle["pred_Ingenuo"].values
    ridge = df_detalle["pred_Ridge"].values

    # Promedio del MLP entre las semillas disponibles
    columnas_mlp = [c for c in df_detalle.columns if c.startswith("pred_MLP_s")]
    if columnas_mlp:
        mlp_prom = df_detalle[columnas_mlp].mean(axis=1).values
    else:
        mlp_prom = ridge

    ax.bar(x - 2 * ancho, reales, ancho, label="Demanda Real", color="#2ca02c")
    ax.bar(x - ancho, ingenuo, ancho, label="Ingenuo (Persistencia)", color="#7f7f7f")
    ax.bar(x, pm4, ancho, label="Promedio Móvil (k=4)", color="#17becf")
    ax.bar(x + ancho, ridge, ancho, label="Regresión Ridge (Lags 1-4)", color="#1f77b4")
    ax.bar(x + 2 * ancho, mlp_prom, ancho, label="MLP PyTorch (Media n=3)", color="#ff7f0e")

    ax.set_title("Figura 2: Pronóstico Mensual Agregado vs Demanda Real (6 Orígenes Móviles)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mes Calendario Objetivo Pronosticado", fontsize=10)
    ax.set_ylabel("Citas Mensuales Totales", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(meses, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida


def generar_figura_3_curvas_consolidadas_mlp(
    historiales: Dict[int, Dict],
    ruta_salida: Union[str, Path]
) -> Path:
    """
    Figura 3: Curvas de entrenamiento y validación de la red neuronal MLP
    consolidadas en un panel multipanel para las semillas 0, 1 y 2.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160, sharey=True)

    for i, s in enumerate([0, 1, 2]):
        ax = axes[i]
        if s in historiales:
            hist = historiales[s]["historial"]
            mejor_ep = historiales[s]["mejor_epoca"]
            epocas = range(1, len(hist["train"]) + 1)

            ax.plot(epocas, hist["train"], color="#1f77b4", linewidth=2, label="Train Loss (MSE)")
            ax.plot(epocas, hist["val"], color="#ff7f0e", linestyle="--", linewidth=2, label="Val Loss Cronológico")

            if mejor_ep <= len(epocas):
                ax.axvline(mejor_ep, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Early Stop (Época {mejor_ep})")

            ax.set_title(f"Semilla {s}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Épocas", fontsize=9)
            if i == 0:
                ax.set_ylabel("Pérdida MSE Normalizado", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Figura 3: Curvas de Entrenamiento y Validación Cronológica Red MLP (PyTorch)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida


def generar_figura_4_error_por_origen(
    df_detalle: pd.DataFrame,
    ruta_salida: Union[str, Path]
) -> Path:
    """
    Figura 4: Error absoluto por origen de evaluación temporal (|Y_real - Y_hat|)
    para comparar la consistencia y estabilidad entre los modelos.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)

    meses = df_detalle["mes_objetivo"].tolist()
    x = range(len(meses))

    err_ing = df_detalle["err_Ingenuo"].values
    err_pm4 = df_detalle["err_Promedio Móvil (k=4)"].values if "err_Promedio Móvil (k=4)" in df_detalle else err_ing
    err_ridge = df_detalle["err_Ridge"].values

    # Error medio del MLP
    cols_err_mlp = [c for c in df_detalle.columns if c.startswith("err_MLP_s")]
    if cols_err_mlp:
        err_mlp = df_detalle[cols_err_mlp].mean(axis=1).values
    else:
        err_mlp = err_ridge

    ax.plot(x, err_ing, marker="s", linewidth=1.8, linestyle="--", color="#7f7f7f", label="Ingenuo")
    ax.plot(x, err_pm4, marker="^", linewidth=2, linestyle="-.", color="#17becf", label="Promedio Móvil (k=4)")
    ax.plot(x, err_ridge, marker="o", linewidth=2.5, color="#1f77b4", label="Regresión Ridge (Lags 1-4)")
    ax.plot(x, err_mlp, marker="D", linewidth=2.2, color="#ff7f0e", label="MLP PyTorch (Promedio n=3)")

    ax.set_title("Figura 4: Error Absoluto de Pronóstico Mensual por Origen de Evaluación", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mes Calendario Evaluado (Origen Temporal)", fontsize=10)
    ax.set_ylabel("Error Absoluto (|Citas Reales - Predicción|)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(meses, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida


# ==============================================================================
# 2. CONSOLIDACIÓN DE RESULTADOS Y TABLA COMPARATIVA GLOBAL
# ==============================================================================

def generar_tabla_comparativa_global(
    df_serie: pd.DataFrame,
    ruta_salida_tabla: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Ejecuta todas las familias de modelos sobre los 6 orígenes y consolida
    la tabla comparativa global con métricas en orden descendente de desempeño.
    """
    # 1. Modelos Estadísticos y Líneas Base
    df_res_est, df_det_est = evaluar_lineas_base_y_estadisticos(df_serie, min_meses_train=3)

    # 2. Red Neuronal MLP PyTorch (3 semillas)
    df_res_mlp, df_det_mlp, historiales = evaluar_mlp_tres_semillas(df_serie, semillas=[0, 1, 2], min_meses_train=3)

    # Unificar filas de resultados
    todas_filas = []

    # Filas de modelos estadísticos y líneas base
    for _, fila in df_res_est.iterrows():
        todas_filas.append(dict(fila))

    # Filas del MLP (solo semillas y media del MLP para no duplicar Ingenuo y Ridge)
    for _, fila in df_res_mlp.iterrows():
        mod_nom = fila["modelo"]
        if "MLP" in mod_nom:
            d = dict(fila)
            d["categoria"] = "Red Neuronal"
            todas_filas.append(d)

    df_global = pd.DataFrame(todas_filas)
    df_global = df_global.sort_values("MAE_mensual").reset_index(drop=True)

    ruta_salida_tabla.parent.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(ruta_salida_tabla, index=False, encoding="utf-8")
    logger.info(f"Tabla comparativa global guardada en: {ruta_salida_tabla}")

    # Fusionar detalles por origen para graficación
    df_detalle_combinado = df_det_mlp.copy()
    if "pred_Promedio Móvil (k=4)" in df_det_est.columns:
        df_detalle_combinado["pred_Promedio Móvil (k=4)"] = df_det_est["pred_Promedio Móvil (k=4)"]
        df_detalle_combinado["err_Promedio Móvil (k=4)"] = df_det_est["err_Promedio Móvil (k=4)"]
    if "pred_ETS (Suavizamiento Exponencial)" in df_det_est.columns:
        df_detalle_combinado["pred_ETS"] = df_det_est["pred_ETS (Suavizamiento Exponencial)"]
        df_detalle_combinado["err_ETS"] = df_det_est["err_ETS (Suavizamiento Exponencial)"]

    return df_global, df_detalle_combinado, historiales


# ==============================================================================
# 3. PIPELINE GENERAL REPRODUCIBLE (UN SOLO COMANDO)
# ==============================================================================

def ejecutar_reporte_completo() -> Dict[str, Union[pd.DataFrame, Path]]:
    """
    Ejecuta el pipeline completo de Fase 6 con un único comando:
    1. Carga la serie procesada.
    2. Evalúa todos los modelos bajo la misma ventana expansiva.
    3. Genera la tabla comparativa global.
    4. Genera las 4 figuras principales.
    5. Preserva las curvas de entrenamiento individuales.
    """
    logger.info("=== INICIANDO GENERACIÓN DE REPORTE REPRODUCIBLE (FASE 6) ===")
    proyecto_root = PROYECTO_ROOT

    ruta_serie = proyecto_root / "data/processed/serie_semanal.csv"
    if not ruta_serie.exists():
        raise FileNotFoundError(
            f"No existe {ruta_serie}. Asegúrese de haber ejecutado la Fase 2."
        )

    df_serie = pd.read_csv(ruta_serie)
    dir_tablas = proyecto_root / "reports/tablas"
    dir_figuras = proyecto_root / "reports/figuras"

    # 1. Tabla comparativa global
    ruta_tabla_global = dir_tablas / "tabla_comparativa_global.csv"
    df_global, df_detalle, historiales = generar_tabla_comparativa_global(
        df_serie=df_serie,
        ruta_salida_tabla=ruta_tabla_global
    )

    # 2. Generación de las 4 Figuras Principales Requeridas
    fig1 = generar_figura_1_serie_completa(
        df_serie=df_serie,
        ruta_salida=dir_figuras / "serie_temporal_completa.png"
    )
    logger.info(f"Figura 1 generada: {fig1}")

    fig2 = generar_figura_2_pronostico_vs_real(
        df_detalle=df_detalle,
        ruta_salida=dir_figuras / "pronostico_mensual_vs_real.png"
    )
    logger.info(f"Figura 2 generada: {fig2}")

    fig3 = generar_figura_3_curvas_consolidadas_mlp(
        historiales=historiales,
        ruta_salida=dir_figuras / "curvas_entrenamiento_mlp.png"
    )
    logger.info(f"Figura 3 generada: {fig3}")

    fig4 = generar_figura_4_error_por_origen(
        df_detalle=df_detalle,
        ruta_salida=dir_figuras / "error_por_origen.png"
    )
    logger.info(f"Figura 4 generada: {fig4}")

    # 3. Conservar las curvas individuales de las 3 semillas
    for s, info in historiales.items():
        ruta_curva_ind = dir_figuras / f"curvas_mlp_semilla{s}.png"
        graficar_curvas_aprendizaje(
            historial=info["historial"],
            semilla=s,
            mejor_epoca=info["mejor_epoca"],
            ruta_salida=ruta_curva_ind
        )

    logger.info("=== REPORTE REPRODUCIBLE GENERADO EXITOSAMENTE ===")
    return {
        "tabla_global": df_global,
        "fig1": fig1,
        "fig2": fig2,
        "fig3": fig3,
        "fig4": fig4
    }


if __name__ == "__main__":
    ejecutar_reporte_completo()
