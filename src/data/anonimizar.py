"""
Módulo de anonimización de datos de citas médicas.
Fase 1: Anonimización (Cumplimiento LOPDP Ecuador).

Este módulo es responsable de leer las exportaciones de datos crudos
(que contienen información personal de pacientes como nombres, cédulas,
teléfonos, diagnósticos clínicos, etc.) y eliminar toda columna identificable,
conservando estricta y exclusivamente:
- fecha_cita
- estado

Genera el archivo anonimizado: data/interim/citas_anonimas.csv
"""

from pathlib import Path
from typing import List, Optional, Union
import argparse
import logging
import pandas as pd
import yaml

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("citas_forecast.anonimizar")

COLUMNAS_OBLIGATORIAS: List[str] = ["fecha_cita", "estado"]


def cargar_configuracion(ruta_config: Optional[Union[str, Path]] = None) -> dict:
    """Carga la configuración centralizada desde config.yaml."""
    if ruta_config is None:
        # Busca config.yaml en la raíz del proyecto
        ruta_config = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    ruta_config = Path(ruta_config)
    if not ruta_config.exists():
        logger.warning(f"No se encontró {ruta_config}. Usando configuración por defecto.")
        return {}
    with open(ruta_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generar_datos_crudos_simulados(
    ruta_salida: Union[str, Path],
    n_registros: int = 650,
    semilla: int = 42
) -> pd.DataFrame:
    """
    Genera un archivo de datos crudos simulados con campos clínicos e identificables.
    Útil cuando los datos reales del consultorio aún no han sido provistos o autorizados,
    garantizando que el proceso de anonimización sea 100% reproducible y verificable.

    Los datos cubren desde septiembre de 2025 en adelante (según cronograma del proyecto).
    """
    import numpy as np

    np.random.seed(semilla)
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # Fechas desde el 1 de septiembre de 2025 por 40 semanas
    fechas_base = pd.date_range(start="2025-09-01", periods=280, freq="D")
    # Filtrar solo días laborables de consulta (lunes a viernes y algunos sábados)
    dias_laborables = [d for d in fechas_base if d.weekday() < 6]

    fechas_citas = []
    estados = []
    nombres = []
    cedulas = []
    telefonos = []
    diagnosticos = []
    notas = []

    posibles_estados = ["atendida", "no_asistida", "cancelada"]
    pesos_estados = [0.75, 0.15, 0.10]

    nombres_ejemplo = [
        "Juan Pérez", "María López", "Carlos Andrade", "Ana Gómez",
        "Luis Torres", "Elena Morales", "Diego Castro", "Sofía Ortiz"
    ]
    diag_ejemplo = ["J00 - Rinofaringitis", "K29 - Gastritis", "M54 - Dorsalgia", "I10 - Hipertensión"]

    for i in range(n_registros):
        dia = np.random.choice(dias_laborables)
        # Formato de fecha YYYY-MM-DD
        fechas_citas.append(pd.to_datetime(dia).strftime("%Y-%m-%d"))
        estados.append(np.random.choice(posibles_estados, p=pesos_estados))
        nombre = np.random.choice(nombres_ejemplo)
        nombres.append(f"{nombre} (Simulado)")
        cedulas.append(f"17{np.random.randint(10000000, 99999999)}")
        telefonos.append(f"09{np.random.randint(10000000, 99999999)}")
        diagnosticos.append(np.random.choice(diag_ejemplo))
        notas.append("Consulta de rutina - Control médico")

    df_raw = pd.DataFrame({
        "id_cita": range(1, n_registros + 1),
        "id_paciente": np.random.randint(100, 500, size=n_registros),
        "nombre_completo": nombres,
        "cedula_identidad": cedulas,
        "telefono_contacto": telefonos,
        "fecha_cita": fechas_citas,
        "estado": estados,
        "codigo_cie10_diagnostico": diagnosticos,
        "observaciones_medicas": notas
    })

    # Ordenar cronológicamente
    df_raw = df_raw.sort_values("fecha_cita").reset_index(drop=True)
    df_raw.to_csv(ruta_salida, index=False, encoding="utf-8")
    logger.info(f"Dataset crudo simulado creado en: {ruta_salida} ({len(df_raw)} registros)")
    return df_raw


def anonimizar_dataframe(
    df: pd.DataFrame,
    columnas_permitidas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aplica las reglas de anonimización estricta sobre un DataFrame.

    Parámetros:
        df: DataFrame original que contiene columnas crudas.
        columnas_permitidas: Lista de columnas a preservar (por defecto ['fecha_cita', 'estado']).

    Retorna:
        DataFrame con única y exclusivamente las columnas especificadas, normalizadas.

    Lanza:
        ValueError: Si faltan columnas obligatorias.
    """
    if columnas_permitidas is None:
        columnas_permitidas = COLUMNAS_OBLIGATORIAS.copy()

    # Normalizar nombres de columnas a minúsculas y sin espacios
    df_trabajo = df.copy()
    df_trabajo.columns = [str(c).strip().lower() for c in df_trabajo.columns]

    # Validar presencia de columnas obligatorias
    faltantes = [c for c in columnas_permitidas if c not in df_trabajo.columns]
    if faltantes:
        raise ValueError(
            f"El DataFrame crudo no contiene las columnas obligatorias requeridas: {faltantes}. "
            f"Columnas disponibles: {list(df_trabajo.columns)}"
        )

    # Filtrar estricta y exclusivamente las columnas permitidas (eliminar cualquier otra)
    df_anonimo = df_trabajo[columnas_permitidas].copy()

    # Normalizar formato de fecha_cita a YYYY-MM-DD
    df_anonimo["fecha_cita"] = pd.to_datetime(df_anonimo["fecha_cita"]).dt.strftime("%Y-%m-%d")

    # Normalizar formato de estado (minúsculas, sin espacios externos)
    df_anonimo["estado"] = df_anonimo["estado"].astype(str).str.strip().str.lower()

    # Ordenar cronológicamente
    df_anonimo = df_anonimo.sort_values("fecha_cita").reset_index(drop=True)

    # Verificación de integridad post-anonimización
    columnas_finales = list(df_anonimo.columns)
    if columnas_finales != columnas_permitidas:
        raise RuntimeError(
            f"Fallo de seguridad LOPDP: las columnas finales {columnas_finales} "
            f"no coinciden exactamente con las permitidas {columnas_permitidas}"
        )

    return df_anonimo


def ejecutar_anonimizacion(
    ruta_entrada: Optional[Union[str, Path]] = None,
    ruta_salida: Optional[Union[str, Path]] = None,
    generar_simulado_si_no_existe: bool = True
) -> Path:
    """
    Flujo principal de la Fase 1. Lee los datos crudos, ejecuta la anonimización
    y guarda el archivo en data/interim/citas_anonimas.csv.

    Retorna:
        Path del archivo CSV anonimizado generado.
    """
    config = cargar_configuracion()
    rutas = config.get("rutas", {})
    datos_cfg = config.get("datos", {})

    cols_permitidas = datos_cfg.get("columnas_anonimizadas", COLUMNAS_OBLIGATORIAS)

    proyecto_root = Path(__file__).resolve().parent.parent.parent

    if ruta_salida is None:
        dir_interim = proyecto_root / rutas.get("interim", "data/interim")
        ruta_salida = dir_interim / "citas_anonimas.csv"
    else:
        ruta_salida = Path(ruta_salida)

    if ruta_entrada is None:
        dir_raw = proyecto_root / rutas.get("raw", "data/raw")
        # Buscar primer CSV en data/raw que no sea gitkeep
        archivos_raw = [f for f in dir_raw.glob("*.csv") if f.is_file()]
        if archivos_raw:
            ruta_entrada = archivos_raw[0]
        else:
            if generar_simulado_si_no_existe:
                ruta_entrada = dir_raw / "export_citas_raw_simulado.csv"
                logger.info(
                    "No se encontró un archivo crudo en data/raw. "
                    "Generando dataset crudo simulado para pruebas de la Fase 1..."
                )
                generar_datos_crudos_simulados(ruta_entrada)
            else:
                raise FileNotFoundError(
                    f"No se encontró ningún archivo CSV crudo en {dir_raw} "
                    "para realizar la anonimización."
                )
    else:
        ruta_entrada = Path(ruta_entrada)

    logger.info(f"Leyendo datos crudos desde: {ruta_entrada}")
    df_raw = pd.read_csv(ruta_entrada, encoding="utf-8")
    num_columnas_originales = len(df_raw.columns)
    columnas_originales = list(df_raw.columns)
    num_filas = len(df_raw)

    logger.info(f"Registros leídos: {num_filas} filas, {num_columnas_originales} columnas.")
    logger.info(f"Columnas detectadas en origen: {columnas_originales}")

    # Ejecutar anonimización
    df_anon = anonimizar_dataframe(df_raw, columnas_permitidas=cols_permitidas)

    # Guardar en destino
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_anon.to_csv(ruta_salida, index=False, encoding="utf-8")

    logger.info(f"Archivo anonimizado guardado exitosamente en: {ruta_salida}")
    logger.info(f"Columnas finales en archivo anonimizado: {list(df_anon.columns)}")
    logger.info(
        f"Se eliminaron {num_columnas_originales - len(df_anon.columns)} columnas sensibles "
        "(nombres, identificadores, diagnósticos, etc.)."
    )

    return ruta_salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1: Anonimización de datos de citas médicas (LOPDP Ecuador)"
    )
    parser.add_argument("--entrada", "-i", type=str, default=None, help="Ruta al archivo crudo CSV")
    parser.add_argument("--salida", "-o", type=str, default=None, help="Ruta de destino del CSV anonimizado")
    parser.add_argument(
        "--generar-simulado",
        action="store_true",
        help="Forzar la generación de un dataset crudo simulado en data/raw/"
    )

    args = parser.parse_args()

    if args.generar_simulado:
        config_gen = cargar_configuracion()
        raw_dir = Path(__file__).resolve().parent.parent.parent / config_gen.get("rutas", {}).get("raw", "data/raw")
        generar_datos_crudos_simulados(raw_dir / "export_citas_raw_simulado.csv")

    ejecutar_anonimizacion(
        ruta_entrada=args.entrada,
        ruta_salida=args.salida,
        generar_simulado_si_no_existe=True
    )
