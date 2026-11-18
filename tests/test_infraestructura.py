"""
Pruebas de infraestructura inicial para la Fase 0 del proyecto Citas Forecast.
Verifica dependencias, estructura de directorios, configuración y reglas de .gitignore.
"""
from pathlib import Path
import pytest
import yaml


def test_importacion_dependencias():
    """Verifica que todas las librerías obligatorias de requirements.txt se importen correctamente."""
    import pandas as pd
    import numpy as np
    import sklearn
    import statsmodels
    import torch
    import matplotlib
    import yaml as pyyaml
    import pytest as pt

    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None
    assert statsmodels.__version__ is not None
    assert torch.__version__ is not None
    assert matplotlib.__version__ is not None
    assert pyyaml.__version__ is not None
    assert pt.__version__ is not None


def test_estructura_directorios():
    """Verifica la existencia de todas las carpetas obligatorias del repositorio."""
    root = Path(__file__).resolve().parent.parent

    directorios_esperados = [
        "data/raw",
        "data/interim",
        "data/processed",
        "src/data",
        "src/modelos",
        "src/evaluacion",
        "tests",
        "reports/tablas",
        "reports/figuras",
        "evidencias/F0_Repositorio",
        "evidencias/F1_Anonimizacion",
        "evidencias/F2_Serie_Semanal",
        "evidencias/F3_Metricas_Validacion",
        "evidencias/F4_Modelos_Estadisticos",
        "evidencias/F5_MLP_PyTorch",
        "evidencias/F6_Reporte_Final",
    ]

    for d in directorios_esperados:
        dir_path = root / d
        assert dir_path.exists(), f"El directorio esperado no existe: {d}"
        assert dir_path.is_dir(), f"La ruta no es un directorio: {d}"


def test_archivos_raiz_obligatorios():
    """Verifica la existencia de los archivos clave en la raíz del repositorio."""
    root = Path(__file__).resolve().parent.parent

    archivos_esperados = [
        "requirements.txt",
        ".gitignore",
        "README.md",
        "config.yaml",
    ]

    for a in archivos_esperados:
        archivo_path = root / a
        assert archivo_path.exists(), f"El archivo esperado no existe: {a}"
        assert archivo_path.is_file(), f"La ruta no es un archivo: {a}"


def test_archivo_config_yaml():
    """Verifica que config.yaml sea un YAML válido y contenga las secciones principales."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "config.yaml no carga un diccionario válido"
    assert "proyecto" in config
    assert "rutas" in config
    assert "datos" in config
    assert "modelos" in config
    assert "evaluacion" in config


def test_gitignore_ignora_datos_sensibles():
    """Verifica que .gitignore contenga reglas explícitas para data/raw y data/interim."""
    root = Path(__file__).resolve().parent.parent
    gitignore_path = root / ".gitignore"

    contenido = gitignore_path.read_text(encoding="utf-8")
    assert "data/raw" in contenido, "data/raw no está contemplado en .gitignore"
    assert "data/interim" in contenido, "data/interim no está contemplado en .gitignore"
