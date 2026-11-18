"""
Módulo de construcción de la serie temporal semanal y análisis exploratorio.
Fase 2: Construcción de la serie semanal (Semana 7).

Transforma los registros médicos anonimizados (data/interim/citas_anonimas.csv)
en una serie temporal regular agregada por semanas calendario (Lunes a Domingo),
aplicando la regla del jueves para la asignación mensual, control de semanas
operativas/cerradas y evaluación del Gate Metodológico de 30 semanas válidas.

Archivos de salida:
- data/processed/serie_semanal.csv (columnas: semana_inicio, citas, operativa, mes_objetivo)
- reports/tablas/dataset.csv (estadísticas descriptivas del dataset)
- reports/figuras/serie_semanal.png (gráfico exploratorio de la serie)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.serie_semanal")

COLUMNAS_OBLIGATORIAS: List[str] = ["semana_inicio", "citas", "operativa", "mes_objetivo"]


def cargar_configuracion(ruta_config: Optional[Union[str, Path]] = None) -> dict:
    """Carga los parámetros desde config.yaml."""
    if ruta_config is None:
        ruta_config = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    ruta_config = Path(ruta_config)
    if not ruta_config.exists():
        logger.warning(f"No se encontró {ruta_config}. Usando configuración por defecto.")
        return {}
    with open(ruta_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def obtener_lunes_semana(fecha: pd.Timestamp) -> pd.Timestamp:
    """Calcula el lunes (inicio de semana ISO) correspondiente a una fecha dada."""
    return fecha.floor("D") - pd.to_timedelta(fecha.weekday(), unit="D")


def asignar_mes_regla_jueves(semana_inicio: pd.Timestamp) -> int:
    """
    Asigna el mes objetivo a una semana según la regla estándar del jueves:
    mes_objetivo = (semana_inicio + 3 días).month
    El mes de la semana corresponde al mes en el que cae su día jueves (mayoría de días).
    """
    jueves = semana_inicio + pd.Timedelta(days=3)
    return int(jueves.month)


def construir_serie_semanal(
    df_citas: pd.DataFrame,
    tipo_demanda: str = "principal",
    semanas_cierre: Optional[List[str]] = None,
    estados_validos: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Agrupa los registros anonimizados en una serie temporal semanal continua.

    Parámetros:
        df_citas: DataFrame con columnas 'fecha_cita' y 'estado'.
        tipo_demanda: 'principal' (atendidas + no asistidas) o 'sensibilidad' (solo atendidas).
        semanas_cierre: Lista de fechas 'YYYY-MM-DD' de inicio de semana donde el consultorio estuvo cerrado.
        estados_validos: Lista opcional de estados considerados válidos.

    Retorna:
        DataFrame con columnas: semana_inicio, citas, operativa, mes_objetivo.
    """
    if semanas_cierre is None:
        semanas_cierre = []
    semanas_cierre_set = {pd.to_datetime(s).strftime("%Y-%m-%d") for s in semanas_cierre}

    # Asegurar copia y tipo datetime
    df = df_citas.copy()
    df["fecha_cita"] = pd.to_datetime(df["fecha_cita"])
    df["estado"] = df["estado"].astype(str).str.strip().str.lower()

    # Selección de definición de cita válida
    if estados_validos is None:
        if tipo_demanda == "sensibilidad":
            estados_validos = ["atendida"]
        else:
            # Demanda principal (planificación de capacidad)
            estados_validos = ["atendida", "no_asistida"]

    logger.info(f"Tipo de demanda: '{tipo_demanda}'. Estados válidos considerados: {estados_validos}")

    # Filtrar solo citas válidas
    df_validas = df[df["estado"].isin(estados_validos)].copy()

    # Calcular semana_inicio (Lunes)
    df_validas["semana_inicio"] = df_validas["fecha_cita"].apply(obtener_lunes_semana)

    # Conteo de citas por semana
    conteo_semanal = df_validas.groupby("semana_inicio").size().to_dict()

    # Determinar rango continuo de semanas completas (de lunes a lunes)
    primera_fecha = df["fecha_cita"].min()
    ultima_fecha = df["fecha_cita"].max()
    primer_lunes = obtener_lunes_semana(primera_fecha)
    ultimo_lunes = obtener_lunes_semana(ultima_fecha)

    rango_semanas = pd.date_range(start=primer_lunes, end=ultimo_lunes, freq="W-MON")

    filas = []
    for sem in rango_semanas:
        sem_str = sem.strftime("%Y-%m-%d")
        mes_obj = asignar_mes_regla_jueves(sem)

        # Verificar si la semana fue de cierre/vacaciones
        if sem_str in semanas_cierre_set:
            operativa = False
            # Si el consultorio estuvo cerrado, las citas son 0
            citas_count = int(conteo_semanal.get(sem, 0))
            if citas_count == 0:
                citas = 0
            else:
                # Si hubo citas registradas en semana marcada cerrada, se respetan pero operativa=False
                citas = citas_count
        else:
            operativa = True
            citas = int(conteo_semanal.get(sem, 0))

        filas.append({
            "semana_inicio": sem_str,
            "citas": citas,
            "operativa": operativa,
            "mes_objetivo": mes_obj
        })

    df_resultado = pd.DataFrame(filas)

    # Ordenar y asegurar tipos
    df_resultado = df_resultado.sort_values("semana_inicio").reset_index(drop=True)
    df_resultado["citas"] = df_resultado["citas"].astype(int)
    df_resultado["operativa"] = df_resultado["operativa"].astype(bool)
    df_resultado["mes_objetivo"] = df_resultado["mes_objetivo"].astype(int)

    return df_resultado[COLUMNAS_OBLIGATORIAS]


def generar_estadisticas_dataset(
    df_serie: pd.DataFrame,
    ruta_salida: Union[str, Path]
) -> pd.DataFrame:
    """
    Calcula y guarda la tabla de estadísticas descriptivas del dataset en formato CSV.
    Campos obligatorios:
    - semanas_totales, semanas_validas, semanas_excluidas, total_citas,
      media, desviacion_estandar, minimo, maximo.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    semanas_totales = int(len(df_serie))
    semanas_validas = int((df_serie["operativa"] == True).sum())
    semanas_excluidas = int((df_serie["operativa"] == False).sum())

    # Las estadísticas de citas se computan sobre las semanas operativas (válidas)
    df_validas = df_serie[df_serie["operativa"] == True]

    if len(df_validas) > 0:
        total_citas = int(df_validas["citas"].sum())
        media = float(round(df_validas["citas"].mean(), 2))
        desviacion_estandar = float(round(df_validas["citas"].std(), 2)) if len(df_validas) > 1 else 0.0
        minimo = int(df_validas["citas"].min())
        maximo = int(df_validas["citas"].max())
    else:
        total_citas = 0
        media = 0.0
        desviacion_estandar = 0.0
        minimo = 0
        maximo = 0

    datos_tabla = [
        {"metrica": "semanas_totales", "valor": semanas_totales},
        {"metrica": "semanas_validas", "valor": semanas_validas},
        {"metrica": "semanas_excluidas", "valor": semanas_excluidas},
        {"metrica": "total_citas", "valor": total_citas},
        {"metrica": "media", "valor": media},
        {"metrica": "desviacion_estandar", "valor": desviacion_estandar},
        {"metrica": "minimo", "valor": minimo},
        {"metrica": "maximo", "valor": maximo},
    ]

    df_stats = pd.DataFrame(datos_tabla)
    df_stats.to_csv(ruta_salida, index=False, encoding="utf-8")
    logger.info(f"Tabla de estadísticas generada en: {ruta_salida}")
    return df_stats


def evaluar_gate_metodologico(semanas_validas: int, umbral: int = 30) -> Dict[str, Union[bool, int, str]]:
    """
    Evalúa el Gate Metodológico de 30 semanas válidas:
    - >= 30 semanas: Se aprueba el uso formal de modelos estadísticos y evaluación de MLP.
    - < 30 semanas: Se activa el PLAN B (líneas base ingenuo y promedio móvil; MLP meramente exploratorio).
    """
    superado = semanas_validas >= umbral
    if superado:
        plan = "Plan A (Modelos estadísticos completos y MLP justificable)"
        dictamen = "APROBADO"
        mensaje = (
            f"GATE METODOLÓGICO SUPERADO: Se dispone de {semanas_validas} semanas válidas "
            f"(umbral mínimo requerido: {umbral}). Se autoriza la comparación formal de métodos "
            f"estadísticos (ETS, SARIMA) y redes neuronales MLP."
        )
    else:
        plan = "Plan B (Líneas base ingenuo y promedio móvil prioritarios; MLP exploratorio)"
        dictamen = "PLAN B ACTIVADO"
        mensaje = (
            f"GATE METODOLÓGICO: ALERTA PLAN B ACTIVADO. Se dispone únicamente de {semanas_validas} "
            f"semanas válidas (< {umbral}). Debido al volumen limitado de datos, se priorizarán "
            f"modelos sencillos y el MLP solo se evaluará de forma exploratoria."
        )

    return {
        "gate_superado": superado,
        "umbral": umbral,
        "semanas_validas": semanas_validas,
        "plan_activo": plan,
        "dictamen": dictamen,
        "mensaje": mensaje
    }


def graficar_serie_semanal(df_serie: pd.DataFrame, ruta_figura: Union[str, Path]) -> Path:
    """Genera y guarda un gráfico exploratorio de la serie temporal semanal."""
    ruta_figura = Path(ruta_figura)
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    x = range(len(df_serie))
    y = df_serie["citas"].values
    operativas = df_serie["operativa"].values

    # Graficar línea de citas
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2, label="Citas válidas semanales")

    # Resaltar semanas no operativas (cerradas)
    no_operativas_idx = [i for i, op in enumerate(operativas) if not op]
    if no_operativas_idx:
        ax.scatter(
            no_operativas_idx,
            [y[i] for i in no_operativas_idx],
            color="#d62728",
            s=100,
            zorder=5,
            label="Semana no operativa (Cierre)"
        )

    # Etiquetas del eje X (cada 2 semanas para legibilidad)
    paso = max(1, len(df_serie) // 15)
    ticks_idx = list(range(0, len(df_serie), paso))
    ax.set_xticks(ticks_idx)
    ax.set_xticklabels([df_serie.loc[i, "semana_inicio"] for i in ticks_idx], rotation=45, ha="right", fontsize=8)

    ax.set_title("Serie Temporal Semanal de Demanda de Citas Médicas (Fase 2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Semana de inicio (Lunes)", fontsize=10)
    ax.set_ylabel("Número de Citas Válidas", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(ruta_figura)
    plt.close()
    logger.info(f"Gráfico de serie semanal guardado en: {ruta_figura}")
    return ruta_figura


def ejecutar_construccion_serie(
    ruta_interim: Optional[Union[str, Path]] = None,
    ruta_salida_serie: Optional[Union[str, Path]] = None,
    ruta_salida_stats: Optional[Union[str, Path]] = None,
    ruta_figura: Optional[Union[str, Path]] = None,
    tipo_demanda: str = "principal"
) -> Tuple[pd.DataFrame, Dict]:
    """Flujo principal de la Fase 2."""
    config = cargar_configuracion()
    rutas = config.get("rutas", {})
    datos_cfg = config.get("datos", {})

    proyecto_root = Path(__file__).resolve().parent.parent.parent

    if ruta_interim is None:
        ruta_interim = proyecto_root / rutas.get("interim", "data/interim") / "citas_anonimas.csv"
    else:
        ruta_interim = Path(ruta_interim)

    if ruta_salida_serie is None:
        ruta_salida_serie = proyecto_root / rutas.get("processed", "data/processed") / "serie_semanal.csv"
    else:
        ruta_salida_serie = Path(ruta_salida_serie)

    if ruta_salida_stats is None:
        ruta_salida_stats = proyecto_root / rutas.get("tablas_reportes", "reports/tablas") / "dataset.csv"
    else:
        ruta_salida_stats = Path(ruta_salida_stats)

    if ruta_figura is None:
        ruta_figura = proyecto_root / rutas.get("figuras_reportes", "reports/figuras") / "serie_semanal.png"
    else:
        ruta_figura = Path(ruta_figura)

    semanas_cierre = datos_cfg.get("semanas_cierre", ["2025-12-29"])
    umbral_gate = datos_cfg.get("min_semanas_validas_gate", 30)

    logger.info(f"Leyendo dataset anonimizado desde: {ruta_interim}")
    df_citas = pd.read_csv(ruta_interim)

    # 1. Construir serie semanal
    df_serie = construir_serie_semanal(
        df_citas=df_citas,
        tipo_demanda=tipo_demanda,
        semanas_cierre=semanas_cierre
    )

    # Guardar serie semanal procesada
    ruta_salida_serie.parent.mkdir(parents=True, exist_ok=True)
    df_serie.to_csv(ruta_salida_serie, index=False, encoding="utf-8")
    logger.info(f"Serie semanal procesada guardada en: {ruta_salida_serie} ({len(df_serie)} semanas)")

    # 2. Generar tabla de estadísticas descriptivas
    generar_estadisticas_dataset(df_serie, ruta_salida_stats)

    # 3. Generar figura de serie temporal
    graficar_serie_semanal(df_serie, ruta_figura)

    # 4. Evaluar Gate Metodológico
    semanas_validas = int((df_serie["operativa"] == True).sum())
    resultado_gate = evaluar_gate_metodologico(semanas_validas, umbral=umbral_gate)

    logger.info(f"--- RESULTADO GATE METODOLÓGICO ---")
    logger.info(resultado_gate["mensaje"])

    return df_serie, resultado_gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 2: Construcción de serie temporal semanal y análisis exploratorio"
    )
    parser.add_argument("--entrada", "-i", type=str, default=None, help="Ruta a citas_anonimas.csv")
    parser.add_argument("--salida", "-o", type=str, default=None, help="Ruta de destino serie_semanal.csv")
    parser.add_argument(
        "--tipo-demanda",
        type=str,
        default="principal",
        choices=["principal", "sensibilidad"],
        help="Definición de cita válida: 'principal' (atendida+no_asistida) o 'sensibilidad' (atendida)"
    )

    args = parser.parse_args()

    ejecutar_construccion_serie(
        ruta_interim=args.entrada,
        ruta_salida_serie=args.salida,
        tipo_demanda=args.tipo_demanda
    )
