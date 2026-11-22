# Citas Forecast

> **Pronóstico de la demanda mensual de citas médicas del siguiente mes a partir de registros semanales mediante modelos de aprendizaje automático**
>
> **Integrantes**: Mateo Sotomayor, Alejandro Moreira  
> **Asignatura**: Inteligencia Artificial 2 — Universidad de Las Américas (UDLA), Quito, Ecuador  
> **Docente**: Prof. Enrique Vinicio Carrera  

---

## 1. Descripción del Proyecto

Proyecto reproducible de ciencia de datos e ingeniería de software para estimar la demanda mensual agregada de citas médicas a partir de series de tiempo semanales en un consultorio médico privado en Quito.

El desarrollo abarca desde la protección de privacidad médica (Ley Orgánica de Protección de Datos Personales - LOPDP), la agregación cronológica mediante la **regla del jueves**, la validación temporal de origen móvil con ventana expansiva (sin fuga de información), hasta la comparación rigurosa entre **líneas base (Ingenuo, Promedio Móvil, Último Mes)**, **modelos estadísticos (Regresión Ridge con lags, ETS, ARIMA)** y una **red neuronal Perceptrón Multicapa (MLP) en PyTorch** evaluada bajo tres semillas aleatorias.

---

## 2. Estructura del Proyecto

```text
citas-forecast/
│
├── data/
│   ├── raw/                  # Datos crudos originales protegidos (LOPDP, gitignored)
│   ├── interim/              # Datos anonimizados (citas_anonimas.csv, gitignored)
│   └── processed/            # Serie temporal semanal consolidada (serie_semanal.csv)
│
├── src/
│   ├── data/
│   │   ├── anonimizar.py     # Fase 1: Filtrado y anonimización estricta LOPDP
│   │   └── construir_serie.py# Fase 2: Construcción semanal y regla del jueves
│   ├── modelos/
│   │   ├── lineas_base.py    # Fase 4: Modelos Ingenuo, Promedio Móvil y Último Mes
│   │   ├── estadisticos.py   # Fase 4: Regresión Ridge con lags, ETS y ARIMA
│   │   └── mlp.py            # Fase 5: Red neuronal MLP en PyTorch con Early Stopping
│   ├── evaluacion/
│   │   ├── metricas.py       # Fase 3: MAE, RMSE, rMAE y WAPE
│   │   ├── origen_movil.py   # Fase 3: Ventana expansiva y escalador sin fuga
│   │   └── generar_reporte.py# Fase 6: Generador automatizado de tablas y figuras
│   └── utils/
│       └── generador_evidencias.py # Generador automático de capturas de evidencias en PNG
│
├── tests/                    # Suite de 45 pruebas automatizadas con pytest
│   ├── test_infraestructura.py
│   ├── test_anonimizacion.py
│   ├── test_serie_semanal.py
│   ├── test_sin_fuga_temporal.py
│   ├── test_modelos_estadisticos.py
│   ├── test_mlp_pytorch.py
│   └── test_reporte_reproducible.py
│
├── reports/
│   ├── tablas/               # Tablas de resultados en CSV
│   │   ├── dataset.csv
│   │   ├── evaluacion_mensual_origenes.csv
│   │   ├── evaluacion_semanal_horizontes.csv
│   │   ├── lineas_base.csv
│   │   ├── resultados_mlp.csv
│   │   └── tabla_comparativa_global.csv
│   └── figuras/              # Gráficos y figuras en PNG
│       ├── serie_temporal_completa.png
│       ├── pronostico_mensual_vs_real.png
│       ├── curvas_entrenamiento_mlp.png
│       ├── error_por_origen.png
│       ├── curvas_mlp_semilla0.png
│       ├── curvas_mlp_semilla1.png
│       └── curvas_mlp_semilla2.png
│
├── evidencias/               # Evidencias PNG organizadas por fases
│   ├── F0_Repositorio/
│   ├── F1_Anonimizacion/
│   ├── F2_Serie_Semanal/
│   ├── F3_Metricas_Validacion/
│   ├── F4_Modelos_Estadisticos/
│   ├── F5_MLP_PyTorch/
│   └── F6_Reporte_Final/
│
├── requirements.txt          # Dependencias del entorno Python
├── .gitignore                # Exclusiones de control de versiones y seguridad
├── config.yaml               # Configuración centralizada de hiperparámetros
└── README.md                 # Documentación técnica completa
```

---

## 3. Instalación y Requisitos

Requisitos: **Python 3.10 o superior** (probado en Python 3.11).

```bash
# 1. Clonar el repositorio
git clone https://github.com/Alejandro-Moreira/IA2-Proyecto.git
cd IA2-Proyecto

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## 4. Reproducción con un Único Comando (Fase 6)

Para regenerar de forma determinista **todas las tablas, modelos y figuras** con un único comando:

```bash
python src/evaluacion/generar_reporte.py
```

### Ejecución paso a paso por fases:

1. **Fase 1: Anonimización (LOPDP)**:
   ```bash
   python src/data/anonimizar.py
   ```
2. **Fase 2: Construcción de la Serie Semanal**:
   ```bash
   python src/data/construir_serie.py
   ```
3. **Fase 3: Validación Temporal de Origen Móvil**:
   ```bash
   python src/evaluacion/origen_movil.py
   ```
4. **Fase 4: Líneas Base y Modelos Estadísticos**:
   ```bash
   python src/modelos/estadisticos.py
   ```
5. **Fase 5: Red Neuronal MLP en PyTorch**:
   ```bash
   python src/modelos/mlp.py
   ```
6. **Fase 6: Reporte Reproducible Global**:
   ```bash
   python src/evaluacion/generar_reporte.py
   ```

---

## 5. Pruebas Automatizadas

El proyecto cuenta con cobertura exhaustiva de pruebas unitarias y de integración que validan la infraestructura, la privacidad LOPDP, el cálculo del calendario, la ausencia de fuga temporal y la consistencia matemática de los modelos:

```bash
pytest -v
```

---

## 6. Resultados Comparativos Globales

Evaluación sobre los **6 orígenes mensuales** mediante ventana expansiva (agregación de semanas al mes objetivo):

| Posición | Modelo | Categoría | MAE mensual | RMSE mensual | rMAE mensual | WAPE | Mejora vs Ingenuo |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Regresión Ridge (Lags 1-4)** | Modelo Estadístico | **9.22** citas | 11.85 | **0.6832** | 0.1366 | **+31.68%** |
| 🥈 **2** | **MLP PyTorch (Semilla 2)** | Red Neuronal | **9.63** citas | 12.33 | **0.7136** | 0.1427 | **+28.64%** |
| 🥉 **3** | **MLP PyTorch (Semilla 1)** | Red Neuronal | **9.90** citas | 11.93 | **0.7335** | 0.1467 | **+26.65%** |
| 4 | **Promedio Móvil ($k=4$)** | Línea Base | **10.00** citas | 10.54 | **0.7407** | 0.1481 | **+25.93%** |
| 5 | **MLP PyTorch (Media $\pm$ Desv, $n=3$)** | Red Neuronal | **10.16** citas | 12.57 | **0.7526** | 0.1505 | **+24.74%** $(\pm 0.69)$ |
| 6 | **ETS (Suavizamiento Exponencial)** | Modelo Estadístico | **10.47** citas | 11.98 | **0.7755** | 0.1551 | **+22.45%** |
| 7 | **ARIMA (Auto-AICc)** | Modelo Estadístico | **10.59** citas | 11.81 | **0.7847** | 0.1569 | **+21.53%** |
| 8 | **MLP PyTorch (Semilla 0)** | Red Neuronal | **10.94** citas | 13.43 | **0.8107** | 0.1621 | **+18.93%** |
| 9 | **Último Mes Completo** | Línea Base | **12.67** citas | 14.82 | **0.9383** | 0.1877 | **+6.17%** |
| 10 | **Ingenuo (Persistencia)** | Línea Base | **13.50** citas | 15.49 | **1.0000** | 0.2000 | +0.00% *(Referencia)* |

### Conclusiones Principales:
1. **Hipótesis 1 comprobada**: Todos los métodos evaluados superan a la persistencia ingenua ($rMAE < 1.0$), logrando reducciones de error de entre el 6% y el 31%.
2. **Hipótesis 3 corroborada**: No se debe presuponer la superioridad de redes complejas. La **Regresión Ridge con Lags 1-4** demostró ser el modelo más preciso, estable y computacionalmente eficiente para el pronóstico de demanda de citas en este entorno clínico.
