"""
Módulo de Red Neuronal Multicapa (MLP) en PyTorch para pronóstico de demanda de citas.
Fase 5: Red Neuronal MLP con PyTorch (Semanas 9-10).

Implementa:
- Arquitectura RedMLP con PyTorch: entradas con lags [1, 2, 3, 4] y promedio móvil (5 features),
  capa oculta (16 neuronas), activación ReLU y regularización L2 (weight_decay).
- Optimización con Adam y Parada Temprana (Early Stopping) con restauración de mejores pesos.
- Partición de validación interna estrictamente cronológica (último 20% de train, sin shuffle).
- Evaluación multi-semilla (semillas 0, 1, 2) reportando media y desviación estándar.
- Pronóstico mensual por estrategia recursiva y comparación directa con Ridge e Ingenuo.
- Generación de curvas de pérdida: reports/figuras/curvas_mlp_semilla{0,1,2}.png
- Reporte tabular de resultados: reports/tablas/resultados_mlp.csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import copy
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Asegurar importación de src
PROYECTO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROYECTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROYECTO_ROOT))

from src.evaluacion.metricas import (
    calcular_mae,
    calcular_rmse,
    calcular_rmae,
    calcular_wape,
    calcular_todas_las_metricas,
)
from src.evaluacion.origen_movil import VentanaExpansiva, EscaladorTemporal
from src.modelos.lineas_base import ModeloIngenuo
from src.modelos.estadisticos import ModeloRegresionLags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.mlp")


# ==============================================================================
# 1. ARQUITECTURA DE LA RED NEURONAL MLP EN PYTORCH
# ==============================================================================

class RedMLP(nn.Module):
    """
    Perceptrón Multicapa (MLP) regularizado para pronóstico univariado con lags.
    Entrada: lags [1, 2, 3, 4] + promedio móvil de 4 semanas (5 dimensiones).
    Oculta : 16 neuronas (o configurable 8-16) con activación ReLU.
    Salida : 1 neurona lineal (predicción semanal).
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 16, output_dim: int = 1):
        super(RedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out


def fijar_semillas(semilla: int = 42):
    """Fija todas las semillas aleatorias para máxima reproducibilidad."""
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


# ==============================================================================
# 2. CONSTRUCCIÓN DE CARACTERÍSTICAS Y PREPROCESAMIENTO TEMPORAL
# ==============================================================================

def construir_dataset_lags(
    serie: np.ndarray,
    lags: List[int] = [1, 2, 3, 4],
    incluir_promedio_movil: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye las matrices de características X y variable objetivo y.
    Features: lags [1, 2, 3, 4] y promedio móvil de los 4 lags.
    """
    max_lag = max(lags)
    X, y = [], []

    for i in range(max_lag, len(serie)):
        feats = [serie[i - l] for l in lags]
        if incluir_promedio_movil:
            feats.append(float(np.mean(feats[:4])))
        X.append(feats)
        y.append(serie[i])

    return np.array(X, dtype=float), np.array(y, dtype=float)


# ==============================================================================
# 3. CLASE ENVOLTORIO PARA ENTRENAMIENTO CON EARLY STOPPING
# ==============================================================================

class EnvoltorioMLP:
    """
    Controla el preprocesamiento, escalado temporal aislado, partición cronológica
    de validación, optimización con Adam, parada temprana y pronóstico recursivo.
    """

    def __init__(
        self,
        semilla: int = 0,
        hidden_dim: int = 16,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        paciencia: int = 20,
        max_epocas: int = 200,
        val_split: float = 0.20
    ):
        self.semilla = semilla
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.paciencia = paciencia
        self.max_epocas = max_epocas
        self.val_split = val_split

        self.escalador_X: Optional[EscaladorTemporal] = None
        self.escalador_y: Optional[EscaladorTemporal] = None
        self.modelo: Optional[RedMLP] = None
        self.historial_perdidas: Dict[str, List[float]] = {"train": [], "val": []}
        self.mejor_epoca: int = 0
        self.ultimo_historial: Optional[np.ndarray] = None

    def fit(self, y_train_serie: np.ndarray) -> "EnvoltorioMLP":
        fijar_semillas(self.semilla)
        arr = np.asarray(y_train_serie, dtype=float)
        self.ultimo_historial = arr.copy()

        # Construir dataset de lags
        X, y = construir_dataset_lags(arr, lags=[1, 2, 3, 4], incluir_promedio_movil=True)
        n_muestras = len(X)
        if n_muestras < 6:
            raise ValueError(f"Historial insuficiente para entrenar MLP ({n_muestras} muestras construidas).")

        # Partición cronológica estricta (último val_split % para validación)
        n_val = max(1, int(n_muestras * self.val_split))
        n_train = n_muestras - n_val

        X_train_raw, y_train_raw = X[:n_train], y[:n_train]
        X_val_raw, y_val_raw = X[n_train:], y[n_train:]

        # Escalado temporal sin fuga (fit exclusivamente sobre X_train_raw e y_train_raw)
        self.escalador_X = EscaladorTemporal()
        self.escalador_y = EscaladorTemporal()

        # Aplanamos para escalador univariado o por columna
        self.escalador_X.fit(X_train_raw.flatten())
        self.escalador_y.fit(y_train_raw)

        X_train_norm = self.escalador_X.transform(X_train_raw.flatten()).reshape(X_train_raw.shape)
        X_val_norm = self.escalador_X.transform(X_val_raw.flatten()).reshape(X_val_raw.shape)

        y_train_norm = self.escalador_y.transform(y_train_raw)
        y_val_norm = self.escalador_y.transform(y_val_raw)

        # Conversión a tensores PyTorch
        t_X_train = torch.tensor(X_train_norm, dtype=torch.float32)
        t_y_train = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(1)
        t_X_val = torch.tensor(X_val_norm, dtype=torch.float32)
        t_y_val = torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(1)

        # Instanciar red
        self.modelo = RedMLP(input_dim=5, hidden_dim=self.hidden_dim, output_dim=1)
        criterio = nn.MSELoss()
        optimizador = optim.Adam(
            self.modelo.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        mejor_val_loss = float("inf")
        mejor_pesos = None
        conteo_sin_mejora = 0
        self.historial_perdidas = {"train": [], "val": []}

        for epoca in range(1, self.max_epocas + 1):
            self.modelo.train()
            optimizador.zero_grad()
            preds_train = self.modelo(t_X_train)
            loss_train = criterio(preds_train, t_y_train)
            loss_train.backward()
            optimizador.step()

            # Validación interna cronológica
            self.modelo.eval()
            with torch.no_grad():
                preds_val = self.modelo(t_X_val)
                loss_val = criterio(preds_val, t_y_val)

            t_loss_val = float(loss_val.item())
            t_loss_train = float(loss_train.item())

            self.historial_perdidas["train"].append(t_loss_train)
            self.historial_perdidas["val"].append(t_loss_val)

            # Early Stopping
            if t_loss_val < mejor_val_loss - 1e-4:
                mejor_val_loss = t_loss_val
                mejor_pesos = copy.deepcopy(self.modelo.state_dict())
                self.mejor_epoca = epoca
                conteo_sin_mejora = 0
            else:
                conteo_sin_mejora += 1
                if conteo_sin_mejora >= self.paciencia:
                    break

        if mejor_pesos is not None:
            self.modelo.load_state_dict(mejor_pesos)

        return self

    def predict_semanal(self, h: int) -> np.ndarray:
        """Pronóstico recursivo semana a semana alimentando las predicciones pasadas."""
        if self.modelo is None or self.ultimo_historial is None:
            raise RuntimeError("Debe llamar a fit() antes de predict_semanal().")

        self.modelo.eval()
        historial = list(self.ultimo_historial)
        predicciones = []

        with torch.no_grad():
            for _ in range(h):
                feats = [historial[-l] for l in [1, 2, 3, 4]]
                feats.append(float(np.mean(feats[:4])))

                feats_arr = np.array(feats, dtype=float).reshape(1, -1)
                feats_norm = self.escalador_X.transform(feats_arr.flatten()).reshape(1, -1)
                t_in = torch.tensor(feats_norm, dtype=torch.float32)

                out_norm = self.modelo(t_in).item()
                out_real = self.escalador_y.inverse_transform(np.array([out_norm]))[0]
                out_real = max(0.0, float(out_real))  # Demanda no negativa

                predicciones.append(out_real)
                historial.append(out_real)

        return np.array(predicciones, dtype=float)

    def predict_mensual(self, num_semanas_mes: int) -> float:
        """Suma de las predicciones semanales hacia el mes objetivo."""
        return float(np.sum(self.predict_semanal(num_semanas_mes)))


# ==============================================================================
# 4. GRAFICACIÓN DE CURVAS DE ENTRENAMIENTO
# ==============================================================================

def graficar_curvas_aprendizaje(
    historial: Dict[str, List[float]],
    semilla: int,
    mejor_epoca: int,
    ruta_salida: Union[str, Path]
) -> Path:
    """Grafica y guarda las curvas de pérdida (Train vs Validation Loss)."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    epocas = range(1, len(historial["train"]) + 1)

    ax.plot(epocas, historial["train"], label="Pérdida Entrenamiento (MSE)", color="#1f77b4", linewidth=2)
    ax.plot(epocas, historial["val"], label="Pérdida Validación Cronológica (MSE)", color="#ff7f0e", linewidth=2, linestyle="--")

    if mejor_epoca <= len(epocas):
        ax.axvline(x=mejor_epoca, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Parada Temprana (Época {mejor_epoca})")

    ax.set_title(f"Curvas de Aprendizaje MLP (PyTorch) - Semilla {semilla}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Épocas de Entrenamiento", fontsize=9)
    ax.set_ylabel("Error Cuadrático Medio (MSE normalizado)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return ruta_salida


# ==============================================================================
# 5. EVALUACIÓN DE LAS TRES SEMILLAS EN LOS 6 ORÍGENES TEMPORALES
# ==============================================================================

def evaluar_mlp_tres_semillas(
    df_serie: pd.DataFrame,
    semillas: List[int] = [0, 1, 2],
    min_meses_train: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict]]:
    """
    Evalúa la red neuronal MLP en las semillas 0, 1 y 2 sobre exactamente
    los mismos 6 orígenes mensuales temporales.
    Compara contra Regresión Ridge e Ingenuo.
    """
    validador = VentanaExpansiva(df_serie)
    y_total = df_serie["citas"].values

    origenes_lista = list(validador.divisiones_mensuales(min_meses_entrenamiento=min_meses_train))
    reales_por_origen = []

    # Diccionario para almacenar errores mensuales por semilla y modelos de referencia
    resultados_origen = []
    historiales_por_semilla = {}

    # Evaluar orígenes para cada modelo
    predicciones = {f"MLP_semilla_{s}": [] for s in semillas}
    predicciones["Ridge"] = []
    predicciones["Ingenuo"] = []

    for idx_orig, ventana in enumerate(origenes_lista):
        num_orig = ventana["origen_num"]
        mes_test = ventana["mes_objetivo"]
        k_semanas = ventana["num_semanas_mes"]
        train_idx = ventana["train_indices"]
        test_idx = ventana["test_indices"]

        y_train = y_total[train_idx]
        y_test = y_total[test_idx]
        real_mes = float(np.sum(y_test))
        if idx_orig >= len(reales_por_origen):
            reales_por_origen.append(real_mes)

        fila_orig = {
            "origen": num_orig,
            "mes_objetivo": mes_test,
            "semanas_train": len(train_idx),
            "semanas_test": k_semanas,
            "real_mensual": real_mes
        }

        # Modelos de referencia
        mod_ing = ModeloIngenuo().fit(y_train)
        pred_ing = mod_ing.predict_mensual(num_semanas_mes=k_semanas)
        predicciones["Ingenuo"].append(pred_ing)
        fila_orig["pred_Ingenuo"] = round(pred_ing, 2)
        fila_orig["err_Ingenuo"] = round(abs(real_mes - pred_ing), 2)

        mod_ridge = ModeloRegresionLags(lags=[1, 2, 3, 4], alpha=1.0).fit(y_train)
        pred_ridge = mod_ridge.predict_mensual(num_semanas_mes=k_semanas)
        predicciones["Ridge"].append(pred_ridge)
        fila_orig["pred_Ridge"] = round(pred_ridge, 2)
        fila_orig["err_Ridge"] = round(abs(real_mes - pred_ridge), 2)

        # Entrenar MLP para cada semilla
        for s in semillas:
            mlp = EnvoltorioMLP(
                semilla=s,
                hidden_dim=16,
                lr=0.01,
                weight_decay=0.001,
                paciencia=20,
                max_epocas=200,
                val_split=0.20
            ).fit(y_train)

            pred_mlp = mlp.predict_mensual(num_semanas_mes=k_semanas)
            predicciones[f"MLP_semilla_{s}"].append(pred_mlp)
            fila_orig[f"pred_MLP_s{s}"] = round(pred_mlp, 2)
            fila_orig[f"err_MLP_s{s}"] = round(abs(real_mes - pred_mlp), 2)

            # Conservar historial de pérdida de la última ventana para graficar curvas
            if idx_orig == len(origenes_lista) - 1:
                historiales_por_semilla[s] = {
                    "historial": mlp.historial_perdidas,
                    "mejor_epoca": mlp.mejor_epoca
                }

        resultados_origen.append(fila_orig)

    df_detalle = pd.DataFrame(resultados_origen)

    # Cálculo de métricas
    reales_arr = np.array(reales_por_origen, dtype=float)
    mae_ingenuo_ref = calcular_mae(reales_arr, predicciones["Ingenuo"])

    resumen_filas = []

    # 1. Ingenuo
    mae_ing = calcular_mae(reales_arr, predicciones["Ingenuo"])
    rmse_ing = calcular_rmse(reales_arr, predicciones["Ingenuo"])
    resumen_filas.append({
        "modelo": "Ingenuo (Persistencia)",
        "MAE_mensual": round(mae_ing, 2),
        "RMSE_mensual": round(rmse_ing, 2),
        "rMAE_mensual": 1.0000,
        "WAPE_mensual": round(calcular_wape(reales_arr, predicciones["Ingenuo"]), 4),
        "mejora_vs_ingenuo": "+0.00%"
    })

    # 2. Ridge
    mae_ridge = calcular_mae(reales_arr, predicciones["Ridge"])
    rmse_ridge = calcular_rmse(reales_arr, predicciones["Ridge"])
    rmae_ridge = calcular_rmae(mae_ridge, mae_ingenuo_ref)
    resumen_filas.append({
        "modelo": "Regresión Ridge (Lags 1-4)",
        "MAE_mensual": round(mae_ridge, 2),
        "RMSE_mensual": round(rmse_ridge, 2),
        "rMAE_mensual": round(rmae_ridge, 4),
        "WAPE_mensual": round(calcular_wape(reales_arr, predicciones["Ridge"]), 4),
        "mejora_vs_ingenuo": f"{(1 - rmae_ridge) * 100:+.2f}%"
    })

    # 3. Cada semilla MLP
    maes_mlp = []
    rmaes_mlp = []
    rmses_mlp = []
    wapes_mlp = []

    for s in semillas:
        preds_s = np.array(predicciones[f"MLP_semilla_{s}"], dtype=float)
        m_mae = calcular_mae(reales_arr, preds_s)
        m_rmse = calcular_rmse(reales_arr, preds_s)
        m_wape = calcular_wape(reales_arr, preds_s)
        m_rmae = calcular_rmae(m_mae, mae_ingenuo_ref)

        maes_mlp.append(m_mae)
        rmaes_mlp.append(m_rmae)
        rmses_mlp.append(m_rmse)
        wapes_mlp.append(m_wape)

        resumen_filas.append({
            "modelo": f"MLP PyTorch (Semilla {s})",
            "MAE_mensual": round(m_mae, 2),
            "RMSE_mensual": round(m_rmse, 2),
            "rMAE_mensual": round(m_rmae, 4),
            "WAPE_mensual": round(m_wape, 4),
            "mejora_vs_ingenuo": f"{(1 - m_rmae) * 100:+.2f}%"
        })

    # 4. Fila Resumen Agregada: MLP Media ± Desv
    media_mae = float(np.mean(maes_mlp))
    std_mae = float(np.std(maes_mlp, ddof=1)) if len(maes_mlp) > 1 else 0.0
    media_rmse = float(np.mean(rmses_mlp))
    media_rmae = float(np.mean(rmaes_mlp))
    std_rmae = float(np.std(rmaes_mlp, ddof=1)) if len(rmaes_mlp) > 1 else 0.0
    media_wape = float(np.mean(wapes_mlp))

    resumen_filas.append({
        "modelo": f"MLP PyTorch (Media ± Desv, n={len(semillas)})",
        "MAE_mensual": round(media_mae, 2),
        "RMSE_mensual": round(media_rmse, 2),
        "rMAE_mensual": round(media_rmae, 4),
        "WAPE_mensual": round(media_wape, 4),
        "mejora_vs_ingenuo": f"{(1 - media_rmae) * 100:+.2f}% (std MAE: ±{std_mae:.2f})"
    })

    df_resumen = pd.DataFrame(resumen_filas).sort_values("MAE_mensual").reset_index(drop=True)
    return df_resumen, df_detalle, historiales_por_semilla


def ejecutar_fase5(
    ruta_serie: Optional[Union[str, Path]] = None,
    ruta_salida_tabla: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flujo principal ejecutable de la Fase 5."""
    proyecto_root = Path(__file__).resolve().parent.parent.parent

    if ruta_serie is None:
        ruta_serie = proyecto_root / "data/processed/serie_semanal.csv"
    else:
        ruta_serie = Path(ruta_serie)

    if ruta_salida_tabla is None:
        ruta_salida_tabla = proyecto_root / "reports/tablas/resultados_mlp.csv"
    else:
        ruta_salida_tabla = Path(ruta_salida_tabla)

    logger.info(f"Cargando serie procesada desde: {ruta_serie}")
    df_serie = pd.read_csv(ruta_serie)

    # 1. Entrenar y evaluar MLP en 3 semillas (0, 1, 2)
    df_resumen, df_detalle, historiales = evaluar_mlp_tres_semillas(
        df_serie=df_serie,
        semillas=[0, 1, 2],
        min_meses_train=3
    )

    # 2. Guardar tabla comparativa de resultados
    ruta_salida_tabla.parent.mkdir(parents=True, exist_ok=True)
    df_resumen.to_csv(ruta_salida_tabla, index=False, encoding="utf-8")
    logger.info(f"Resultados del MLP guardados en: {ruta_salida_tabla}")

    # 3. Generar figuras de curvas de aprendizaje para cada semilla
    dir_figuras = proyecto_root / "reports/figuras"
    for s, info in historiales.items():
        ruta_fig = dir_figuras / f"curvas_mlp_semilla{s}.png"
        graficar_curvas_aprendizaje(
            historial=info["historial"],
            semilla=s,
            mejor_epoca=info["mejor_epoca"],
            ruta_salida=ruta_fig
        )
        logger.info(f"Curvas de pérdida guardadas para semilla {s}: {ruta_fig}")

    return df_resumen, df_detalle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 5: Red Neuronal MLP con PyTorch")
    parser.add_argument("--serie", type=str, default=None, help="Ruta a serie_semanal.csv")
    args = parser.parse_args()
    ejecutar_fase5(ruta_serie=args.serie)
