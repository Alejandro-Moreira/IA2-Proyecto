# Proyecto: Pronóstico de la Demanda Mensual de Citas Médicas a partir de Registros Semanales mediante Modelos de Aprendizaje Automático

**Asignatura:** Inteligencia Artificial II  
**Institución:** Universidad de Las Américas (UDLA) — Ecuador  
**Autores:** Mateo Sotomayor, Alejandro Moreira  
**Repositorio GitHub:** [https://github.com/Alejandro-Moreira/IA2-Proyecto.git](https://github.com/Alejandro-Moreira/IA2-Proyecto.git)  
**Entorno de Ejecución:** Python 3.11, PyTorch 2.6, Statsmodels, Scikit-learn, Pandas  

---

## 1. Resumen Ejecutivo y Metodología General

El objetivo primordial del proyecto es desarrollar, evaluar y documentar un sistema integral de aprendizaje automático para el **pronóstico de la demanda mensual de citas médicas efectivas** del siguiente mes a partir de agregaciones semanales, garantizando:
1. **Cumplimiento estricto de la LOPDP ecuatoriana**: Cero almacenamiento o fuga de identificadores personales de pacientes en el repositorio de código.
2. **Ausencia total de fuga temporal (Data Leakage)**: Estricto orden cronológico en la construcción de características, ajuste de escaladores de normalización únicamente en conjuntos de entrenamiento, y evaluación mediante origen móvil con ventana expansiva.
3. **Reproducibilidad integral**: Regeneración de todos los experimentos, tablas y figuras mediante un único comando (`python src/evaluacion/generar_reporte.py`).
4. **Validación empírica y teórica**: Contraste riguroso entre tres familias de modelos (Líneas base heurísticas, Modelos estadísticos y Redes neuronales profundas en PyTorch).

```mermaid
flowchart TD
    subgraph S1["Fase 1: Privacidad & LOPDP"]
        D1["Datos Crudos (data/raw)"] -->|Filtrado estricto LOPDP| D2["Datos Anonimizados (data/processed)"]
    end
    subgraph S2["Fase 2: Agregación Temporal"]
        D2 -->|Lunes inicio + Regla del Jueves| D3["Serie Semanal (40 semanas, 39 operativas)"]
        D3 -->|Gate Metodológico: 39 >= 30| G1["Aprobado: Plan A (Horizonte Mensual)"]
    end
    subgraph S3["Fase 3: Validación Temporal"]
        G1 -->|Ventana Expansiva 6 Orígenes| V1["Cortes Enero - Junio 2026"]
        V1 -->|Escalador Temporal ajustado solo en Train| V2["Entrenamiento / Validación / Test"]
    end
    subgraph S4["Fases 4 y 5: Modelado"]
        V2 --> M1["Líneas Base (Ingenuo, Media Móvil, Último Mes)"]
        V2 --> M2["Modelos Estadísticos (Ridge, ETS, ARIMA Auto-AICc)"]
        V2 --> M3["Red Neuronal MLP PyTorch (Multi-semilla + Early Stopping)"]
    end
    subgraph S5["Fase 6: Resultados & Reportes"]
        M1 & M2 & M3 --> R1["Tabla Comparativa Global (MAE, RMSE, rMAE, WAPE)"]
        R1 --> R2["Figuras Científicas & Reporte Reproducible"]
    end
```

---

## 2. Fase 0: Repositorio e Infraestructura

### 2.1 Descripción de la Fase
En esta fase se estructuró la arquitectura de software siguiendo las mejores prácticas de ingeniería para proyectos de ciencia de datos, organizando las carpetas modulares (`src/data/`, `src/evaluacion/`, `src/modelos/`, `src/utils/`, `reports/`, `tests/`, `evidencias/`). Se definieron las dependencias en `requirements.txt` y se configuró el archivo `.gitignore` para bloquear el seguimiento de `data/raw/*` y `data/interim/*`, asegurando el cumplimiento legal desde el primer commit.

### 2.2 Capturas de Pantalla de Ejecución
A continuación se evidencian la inicialización de la estructura de directorios, la instalación del entorno virtual y la ejecución inicial de pruebas de infraestructura:

![Estructura del Repositorio](evidencias/F0_Repositorio/F0_01_estructura_repositorio.png)
*Figura 2.1: Inspección de directorios iniciales y archivos raíz del proyecto (`config.yaml`, `requirements.txt`, `.gitignore`).*

![Dependencias Instaladas](evidencias/F0_Repositorio/F0_02_dependencias_instaladas.png)
*Figura 2.2: Verificación de paquetes instalados en el entorno virtual (`torch`, `statsmodels`, `scikit-learn`, `pytest`).*

![Pruebas de Infraestructura Aprobadas](evidencias/F0_Repositorio/F0_03_pytest_exitoso.png)
*Figura 2.3: Validación mediante pytest de la estructura física y configuración inicial.*

---

## 3. Fase 1: Anonimización de Datos y Cumplimiento Normativo (LOPDP)

### 3.1 Marco Normativo y Regla de Filtrado
En concordancia con el Art. 34 y conexos de la Ley Orgánica de Protección de Datos Personales (LOPDP) del Ecuador, ningún dato clínico o identificable puede residir en repositorios públicos. El módulo `src/data/anonimizar.py` procesa los registros de citas eliminando de forma irreversible las 7 columnas sensibles (`id_paciente`, `nombre`, `cedula`, `telefono`, `correo`, `direccion`, `diagnostico`), reteniendo única y exclusivamente:
- `fecha_cita`: Fecha del evento médico (formato `YYYY-MM-DD`).
- `estado`: Estado de atención (`ATENDIDO`, `CANCELADO`, `NO_ASISTIO`).

### 3.2 Capturas de Pantalla de Ejecución
La ejecución valida que los datos procesados en memoria o guardados en `data/processed/` carezcan totalmente de trazas personales:

![Anonimización Exitosa](evidencias/F1_Anonimizacion/F1_01_anonimizacion_exitosa.png)
*Figura 3.1: Ejecución del proceso de anonimización irreversible y generación de datos intermedios.*

![Validación de Columnas LOPDP](evidencias/F1_Anonimizacion/F1_02_validacion_columnas.png)
*Figura 3.2: Pruebas unitarias comprobando la presencia estricta de 2 columnas y cero fugas de PII.*

---

## 4. Fase 2: Construcción de la Serie Semanal y Análisis Exploratorio

### 4.1 Definición Operativa y Regla del Jueves
La serie temporal se construye agregando los registros por semana con inicio en lunes (`semana_inicio`). La demanda efectiva de atención se define estrictamente como citas con `estado == 'ATENDIDO'`.
Para la asignación del mes calendario al que pertenece cada semana, se implementó la **Regla del Jueves** (norma ISO 8601): una semana pertenece al mes que contenga el día jueves de dicha semana.

Adicionalmente, se identifican las semanas no operativas (por ejemplo, la semana del 2025-12-29 correspondiente al receso de fin de año del centro médico), marcando la bandera booleana `operativa = False`.

### 4.2 Evaluación del Gate Metodológico
- **Total semanas registradas:** 40 semanas (Septiembre 2025 a Junio 2026).
- **Semanas operativas:** 39 semanas.
- **Criterio del Gate:** Se requieren $\ge 30$ semanas operativas para sostener el Plan A (pronóstico mensual agregado vía semanas).
- **Resultado:** **Gate Aprobado** ($39 \ge 30$).

### 4.3 Gráfica de la Serie Semanal Generada
La evolución histórica de citas atendidas muestra el comportamiento de la demanda y la semana de receso:

![Serie Semanal de Citas Médicas](reports/figuras/serie_semanal.png)
*Figura 4.1: Serie temporal semanal de citas atendidas con marcación de cierre operativo.*

### 4.4 Capturas de Pantalla de Ejecución

![Serie Semanal Generada](evidencias/F2_Serie_Semanal/F2_01_serie_semanal_generada.png)
*Figura 4.2: Ejecución del módulo `construir_serie.py` y resumen de registros procesados.*

![Tabla de Estadísticas](evidencias/F2_Serie_Semanal/F2_02_tabla_estadisticas_dataset.png)
*Figura 4.3: Exportación de métricas estadísticas descriptivas a `reports/tablas/dataset.csv`.*

![Gate Metodológico Aprobado](evidencias/F2_Serie_Semanal/F2_03_gate_metodologico_30_semanas.png)
*Figura 4.4: Verificación automatizada de superación del umbral de 30 semanas operativas.*

---

## 5. Fase 3: Métricas y Esquema de Validación Temporal (Sin Fuga)

### 5.1 Métricas Formales de Evaluación
Para cuantificar el desempeño se implementaron métricas orientadas tanto a precisión estadística como a interpretabilidad en gestión de salud:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

$$\text{rMAE} = \frac{\text{MAE}_{\text{modelo}}}{\text{MAE}_{\text{ingenuo}}} \quad (\text{rMAE} < 1.0 \implies \text{Supera a la persistencia})$$

$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$

### 5.2 Esquema de Origen Móvil (Ventana Expansiva)
Se definieron 6 orígenes de corte mensual que corresponden a los meses de **Enero a Junio de 2026**. Para cada origen $T$:
- El conjunto de entrenamiento contiene exclusivamente las semanas $t \le T$.
- El horizonte de prueba corresponde a las semanas del mes calendario objetivo ($h \in [4, 5]$ semanas).
- El escalador `EscaladorTemporal` calcula $\min$ y $\max$ únicamente sobre el bloque de entrenamiento y se reajusta en cada corte.
- El pronóstico mensual total se calcula sumando los pronósticos semanales del mes: $\hat{Y}_{\text{mes}} = \sum_{w \in \text{mes}} \hat{y}_w$.

### 5.3 Capturas de Pantalla de Ejecución

![Métricas de Evaluación](evidencias/F3_Metricas_Validacion/F3_01_metricas_evaluacion.png)
*Figura 5.1: Implementación y comprobación de fórmulas matemáticas de métricas.*

![Ventana Expansiva](evidencias/F3_Metricas_Validacion/F3_02_ventana_expansiva.png)
*Figura 5.2: Desglose de los 6 orígenes móviles temporales y cálculo de horizontes semanales.*

![Pruebas Sin Fuga Temporal](evidencias/F3_Metricas_Validacion/F3_03_pruebas_sin_fuga_temporal.png)
*Figura 5.3: Suite de 10 pruebas unitarias verificando la inmunidad a datos del futuro.*

---

## 6. Fase 4: Líneas Base y Modelos Estadísticos

### 6.1 Modelos Implementados
Se evaluaron seis métodos en cada una de las 6 ventanas de corte:
1. **Ingenuo (Persistencia):** Repite el último valor semanal observado para todas las semanas del horizonte.
2. **Promedio Móvil ($k=4$):** Media aritmética de las 4 semanas operativas más recientes.
3. **Último Mes Completo:** Repite la suma del mes calendario inmediatamente anterior.
4. **Regresión Ridge:** Regresión lineal con regularización $L_2$ ($\alpha=1.0$), usando como características los rezagos $t-1, t-2, t-3, t-4$ y el promedio móvil de 4 periodos.
5. **ETS (Suavizamiento Exponencial):** Modelo Holt/Winters no estacional con selección de nivel y tendencia aditiva amortiguada.
6. **ARIMA (Auto-AICc):** Búsqueda de órdenes $(p, d, q)$ optimizando el criterio de información corregido de Akaike exclusivamente sobre el bloque de entrenamiento.

### 6.2 Capturas de Pantalla de Ejecución

![Modelos Implementados](evidencias/F4_Modelos_Estadisticos/F4_01_modelos_implementados.png)
*Figura 6.1: Inspección de las clases de líneas base y modelos estadísticos en `src/modelos/`.*

![Ejecución de Modelos Estadísticos](evidencias/F4_Modelos_Estadisticos/F4_02_ejecucion_modelos.png)
*Figura 6.2: Ejecución secuencial sobre los 6 orígenes móviles de evaluación.*

![Tabla Comparativa Líneas Base](evidencias/F4_Modelos_Estadisticos/F4_03_tabla_comparativa_lineas_base.png)
*Figura 6.3: Comparación inicial de errores MAE, RMSE y rMAE entre métodos estadísticos y heurísticos.*

---

## 7. Fase 5: Red Neuronal MLP en PyTorch

### 7.1 Arquitectura y Entrenamiento
Se construyó un Perceptrón Multicapa (`RedMLP`) adaptado a series cortas para mitigar el sobreajuste:
- **Capa de entrada:** 5 neuronas (Lags 1 a 4 + Promedio móvil de 4 semanas).
- **Capa oculta:** 16 neuronas con función de activación ReLU.
- **Capa de salida:** 1 neurona (pronóstico de demanda semanal).
- **Optimizador:** Adam ($\text{lr} = 0.01$) con decaimiento de pesos $L_2$ (`weight_decay = 0.001`).
- **Función de pérdida:** Error cuadrático medio (MSELoss).
- **Early Stopping:** Monitoreo sobre el último 20% cronológico del bloque de entrenamiento con paciencia de 20 épocas.
- **Evaluación Multi-semilla:** Se ejecutó con semillas aleatorias 0, 1 y 2 para reportar media y desviación estándar ($\mu \pm \sigma$).

### 7.2 Curvas de Aprendizaje (Entrenamiento vs. Validación)
A continuación se muestran las curvas de pérdida para cada una de las semillas evaluadas, ilustrando la estabilidad de la convergencia:

| Semilla 0 | Semilla 1 | Semilla 2 |
| :---: | :---: | :---: |
| ![MLP Semilla 0](reports/figuras/curvas_mlp_semilla0.png) | ![MLP Semilla 1](reports/figuras/curvas_mlp_semilla1.png) | ![MLP Semilla 2](reports/figuras/curvas_mlp_semilla2.png) |

*Figura 7.1: Curvas de convergencia de MSE de entrenamiento y validación para semillas 0, 1 y 2.*

### 7.3 Capturas de Pantalla de Ejecución

![Arquitectura MLP](evidencias/F5_MLP_PyTorch/F5_01_arquitectura_mlp.png)
*Figura 7.2: Inspección del módulo `src/modelos/mlp.py` y arquitectura `RedMLP` en PyTorch.*

![Entrenamiento y Curvas](evidencias/F5_MLP_PyTorch/F5_02_entrenamiento_curvas.png)
*Figura 7.3: Proceso de entrenamiento iterativo con guardado automático de curvas por semilla.*

![Comparativa MLP vs Ridge vs Ingenuo](evidencias/F5_MLP_PyTorch/F5_03_tabla_comparativa_mlp_ridge_ingenuo.png)
*Figura 7.4: Tabla de desempeño multi-semilla confirmando estabilidad numérica.*

---

## 8. Fase 6: Reporte Reproducible y Visualizaciones Finales

### 8.1 Resultados Globales Consolidados
Al integrar todos los modelos evaluados de forma idéntica en los 6 orígenes móviles mensuales de prueba, se obtiene la tabla comparativa global:

| Modelo | Categoría | MAE Mensual | RMSE Mensual | rMAE Mensual | WAPE Mensual | Mejora vs. Ingenuo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Ridge (Lags 1-4)** | **Modelo Estadístico** | **9.22** | **11.85** | **0.6832** | **0.1366** | **+31.68%** |
| MLP PyTorch (Semilla 2) | Red Neuronal | 9.63 | 12.33 | 0.7136 | 0.1427 | +28.64% |
| MLP PyTorch (Semilla 1) | Red Neuronal | 9.90 | 11.93 | 0.7335 | 0.1467 | +26.65% |
| Promedio Móvil ($k=4$) | Línea Base | 10.00 | 10.54 | 0.7407 | 0.1481 | +25.93% |
| **MLP PyTorch (Media $\pm$ Desv, $n=3$)** | **Red Neuronal** | **10.16** | **12.57** | **0.7526** | **0.1505** | **+24.74% ($\pm$0.69)** |
| ETS (Suavizamiento Exponencial) | Modelo Estadístico | 10.47 | 11.98 | 0.7755 | 0.1551 | +22.45% |
| ARIMA (Auto-AICc) | Modelo Estadístico | 10.59 | 11.81 | 0.7847 | 0.1569 | +21.53% |
| MLP PyTorch (Semilla 0) | Red Neuronal | 10.94 | 13.43 | 0.8107 | 0.1621 | +18.93% |
| Último Mes Completo | Línea Base | 12.67 | 14.82 | 0.9383 | 0.1877 | +6.17% |
| Ingenuo (Persistencia) | Línea Base | 13.50 | 15.49 | 1.0000 | 0.2000 | +0.00% |

---

### 8.2 Las 4 Figuras Científicas Principales

#### Figura I: Serie Temporal Histórica y Períodos Operativos
![Serie Temporal Completa](reports/figuras/serie_temporal_completa.png)
*Figura 8.1: Comportamiento semanal histórico de citas atendidas, corte de entrenamiento y receso de fin de año.*

#### Figura II: Pronóstico Mensual vs. Demanda Real Observada
![Pronóstico Mensual vs Real](reports/figuras/pronostico_mensual_vs_real.png)
*Figura 8.2: Comparación mes a mes de las predicciones de cada modelo frente al total real de citas atendidas en los 6 orígenes de evaluación.*

#### Figura III: Curvas de Entrenamiento y Early Stopping de la Red MLP
![Curvas Entrenamiento MLP](reports/figuras/curvas_entrenamiento_mlp.png)
*Figura 8.3: Evolución de las funciones de pérdida y punto de detención temprana de la red neuronal.*

#### Figura IV: Distribución del Error Absoluto por Origen Móvil
![Error por Origen](reports/figuras/error_por_origen.png)
*Figura 8.4: Análisis de robustez y varianza del error en cada corte temporal (Enero a Junio 2026).*

---

### 8.3 Capturas de Pantalla de Ejecución de la Fase 6

![Ejecución del Generador de Reportes](evidencias/F6_Reporte_Final/F6_01_ejecucion_generador_reportes.png)
*Figura 8.5: Consola de ejecución del comando único de reporte reproducible `generar_reporte.py`.*

![Tablas Finales Consolidadas](evidencias/F6_Reporte_Final/F6_02_tablas_finales.png)
*Figura 8.6: Visualización de las tablas de datos generadas en `reports/tablas/`.*

![Figuras Generadas](evidencias/F6_Reporte_Final/F6_03_figuras_generadas.png)
*Figura 8.7: Confirmación de archivos de figuras PNG de alta resolución.*

![Estructura Final del Proyecto](evidencias/F6_Reporte_Final/F6_04_estructura_final_proyecto.png)
*Figura 8.8: Árbol completo de directorios del proyecto Citas Forecast.*

![README con Instrucciones de Reproducción](evidencias/F6_Reporte_Final/F6_05_readme_instrucciones.png)
*Figura 8.9: Documentación de despliegue y reproducción en `README.md`.*

![Suite Completa de 45 Pruebas Exitosas](evidencias/F6_Reporte_Final/F6_06_ejecucion_completa_sin_errores.png)
*Figura 8.10: Salida del comando `pytest tests/ -v` con 45 pruebas unitarias aprobadas.*

---

## 9. Discusión Científica y Conclusiones

1. **Superioridad de los Modelos Lineales Regularizados en Muestras Pequeñas**:  
   El modelo de **Regresión Ridge** alcanzó el mejor rendimiento general con un $\text{MAE} = 9.22$ y un $\text{rMAE} = 0.6832$ (una reducción de error del 31.68% respecto a la línea base ingenua). Este resultado concuerda con la literatura empírica (Makridakis et al., 2018; Jones et al., 2008): en series temporales del sector salud con tamaños de muestra limitados (39 semanas operativas), la regularización analítica $L_2$ previene el sobreajuste de forma más eficiente que las arquitecturas neuronales de alta parametrización.

2. **Rendimiento Competitivo del MLP PyTorch**:  
   La red MLP con arquitectura compacta (5-16-1), regularización de pesos y Early Stopping demostró ser un predictor altamente competitivo, alcanzando un $\text{MAE}$ medio de $10.16 \pm 0.69$ y logrando con la semilla 2 un error de $9.63$ citas. La varianza entre semillas ($\sigma = 0.69$) es moderada y confirma la importancia de reportar resultados sobre múltiples inicializaciones estocásticas.

3. **Inadecuación de Modelos Complejos sin Suficiente Historia**:  
   Los modelos ARIMA y ETS obtuvieron un $\text{MAE}$ de $10.59$ y $10.47$ respectivamente, viéndose superados incluso por el Promedio Móvil ($k=4$, $\text{MAE} = 10.00$). Esto se atribuye a la sensibilidad de la estimación de parámetros autorregresivos ante la presencia de semanas atípicas (como cierres de fin de año) en series menores a 50 observaciones.

4. **Calidad de Ingeniería de Software y Reproducibilidad**:  
   El proyecto cuenta con **45 pruebas unitarias automatizadas** que garantizan que el código es inmune a fugas de datos futuros, cumple los requerimientos de la LOPDP, y puede ser reproducido de principio a fin con un solo comando.

---

## 10. Instrucciones de Reproducción Rápida

Para replicar todos los experimentos y generar las tablas y figuras:

```bash
# 1. Clonar el repositorio
git clone https://github.com/Alejandro-Moreira/IA2-Proyecto.git
cd IA2-Proyecto

# 2. Crear y activar entorno virtual
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la suite completa de pruebas unitarias (45 tests)
pytest tests/ -v

# 5. Generar reporte completo, tablas y figuras con un solo comando
python src/evaluacion/generar_reporte.py
```
