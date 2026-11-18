# Citas Forecast

Proyecto académico de ciencia de datos para el pronóstico de demanda semanal y agregación mensual de citas médicas mediante modelos estadísticos y redes neuronales en PyTorch.

## Estructura del Proyecto

```text
citas-forecast/
│
├── data/
│   ├── raw/                  # Datos crudos originales (protegidos / gitignored)
│   ├── interim/              # Datos anonimizados LOPDP (gitignored)
│   └── processed/            # Series temporales consolidadas
│
├── src/
│   ├── data/                 # Anonimización y construcción de series
│   ├── modelos/              # Líneas base, modelos estadísticos y MLP PyTorch
│   └── evaluacion/           # Métricas, origen móvil y generador de reportes
│
├── tests/                    # Pruebas automatizadas con pytest
├── reports/
│   ├── tablas/               # Tablas de resultados en CSV
│   └── figuras/              # Gráficos y curvas en PNG
│
├── evidencias/               # Evidencias organizadas por fases
│   ├── F0_Repositorio/
│   ├── F1_Anonimizacion/
│   ├── F2_Serie_Semanal/
│   ├── F3_Metricas_Validacion/
│   ├── F4_Modelos_Estadisticos/
│   ├── F5_MLP_PyTorch/
│   └── F6_Reporte_Final/
│
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Reglas de exclusión de git
├── README.md                 # Documentación del proyecto
└── config.yaml               # Configuración centralizada
```

## Instalación y Entorno

1. Clonar o acceder a la carpeta del proyecto:
   ```bash
   cd citas-forecast
   ```

2. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar las pruebas automatizadas:
   ```bash
   pytest
   ```
