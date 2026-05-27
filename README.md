<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/188/188905.png" alt="FIFA World Cup 2026" width="100"/>
</p>

<h1 align="center">⚽ IA Predictor — FIFA World Cup 2026</h1>

<p align="center">
  <strong>Simulador predictivo avanzado del Mundial 2026 impulsado por Machine Learning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2-FF6600?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Plotly-6.7-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

<p align="center">
  <a href="#-descripción">Descripción</a> •
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-arquitectura">Arquitectura</a> •
  <a href="#-instalación">Instalación</a> •
  <a href="#-uso">Uso</a> •
  <a href="#-metodología">Metodología</a> •
  <a href="#-estructura-del-proyecto">Estructura</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 📖 Descripción

**IA Predictor — Mundial 2026** es una aplicación interactiva de ciencia de datos que simula el desarrollo completo de la Copa del Mundo FIFA 2026. Utiliza un modelo **XGBoost** con clase balanceada, entrenado con más de **5.000 partidos internacionales (2010–2025)**, combinado con simulaciones **Monte Carlo Tree Search (MCTS)** para estimar probabilidades de clasificación, avance en rondas eliminatorias y proclamar un campeón.

La interfaz, construida con **Streamlit**, presenta visualizaciones profesionales con diseño *dark-mode*, *glassmorphism*, animaciones y gráficos interactivos hechos con **Plotly `graph_objects`** para un aspecto de calidad *data-science & editorial*.

---

## ✨ Features

### 📊 Fase de Grupos
- Probabilidades de clasificación por grupo tras **10.000 simulaciones MCTS**
- Gráficos de barras horizontales con gradientes de color dinámicos
- KPI cards con el favorito y el equipo con menor probabilidad
- Línea de referencia al 50% como umbral visual

### ⚔️ Simulador de Rondas KO
- **Butterfly / Tornado Charts** para cada ronda eliminatoria (R32 → Semifinales)
- **Donut Ring Chart** para la Gran Final con indicador de campeón
- Métricas de resumen: diferencia media, partido más reñido, mayor favorito
- Soporte completo: Dieciseisavos → Octavos → Cuartos → Semifinales → Final

### 🔬 Análisis Técnico del Modelo
| Visualización | Descripción |
|---|---|
| **Curva ROC Multiclase** | Discriminación por clase (Victoria / Empate / Derrota) con AUC por categoría |
| **Feature Importance** | Importancia relativa de las 5 features del modelo (Gain score) |
| **Evolución ELO Histórica** | Series temporales interactivas del Top 8 selecciones (2010–2025) |
| **Matriz de Correlación** | Heatmap triangular entre features con escala divergente |
| **Radar Chart Comparativo** | Perfil multidimensional de cualquier par de selecciones |
| **Distribución Histórica** | Tendencia anual de resultados + donut de distribución global |

### 🎨 Diseño & UX
- Dark theme profesional con paleta **Indigo / Violet / Emerald**
- Componentes *glassmorphism* con `backdrop-filter`
- Micro-animaciones CSS (`fadeInUp`, `pulse`)
- Tipografía [Inter](https://fonts.google.com/specimen/Inter) vía Google Fonts
- Marca de agua *"IA Predictor 2026"* en cada gráfica
- Totalmente responsivo

---

## 🖼️ Demo

> **Ejecuta la app localmente** para explorar todas las visualizaciones interactivas.

```bash
streamlit run app.py
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                     STREAMLIT UI                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  Fase de  │  │  Rondas KO   │  │  Análisis Modelo  │ │
│  │  Grupos   │  │  (Butterfly / │  │  (ROC, Radar,     │ │
│  │  (Barras) │  │   Donut)     │  │   Correlación)    │ │
│  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘ │
│       │               │                   │             │
│       └───────────────┼───────────────────┘             │
│                       │                                 │
│              ┌────────▼────────┐                        │
│              │  PLOTLY G.O.    │                        │
│              │  Visualizations │                        │
│              └────────┬────────┘                        │
└───────────────────────┼─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  ┌─────▼─────┐  ┌──────▼──────┐  ┌────▼─────┐
  │  XGBoost   │  │  MCTS       │  │  Feature  │
  │  Model     │  │  Simulation │  │  Engine   │
  │ (.pkl)     │  │  (10K iter) │  │  (ELO,MV) │
  └─────┬─────┘  └──────┬──────┘  └────┬─────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
            ┌───────────▼───────────┐
            │   DATA LAYER          │
            │  results.csv (1872+)  │
            │  national_teams.csv   │
            │  df_train.csv (5011)  │
            │  df_mundial.csv (48)  │
            └───────────────────────┘
```

---

## 🚀 Instalación

### Requisitos previos
- **Python** ≥ 3.10
- **pip** (gestor de paquetes)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<vicentenr8>/SIM_MUNDIAL2026.git
cd SIM_MUNDIAL2026

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## ▶️ Uso

```bash
# Iniciar la aplicación
streamlit run app.py
```

La app se abrirá por defecto en `http://localhost:8501`. Usa el menú lateral para navegar entre las tres secciones principales.

---

## 🧠 Metodología

### Pipeline de Machine Learning

```
Datos crudos (1872–2025)
    │
    ▼
Feature Engineering
    ├── ELO Rating diferencial
    ├── Market Value (log-scaled)
    ├── Momentum 15 (últimos 15 partidos)
    ├── Knockout Performance (rendimiento en eliminatorias)
    └── Pressure Index (presión competitiva)
    │
    ▼
XGBoost Classifier
    ├── Balanced class weights
    ├── 3 clases: Victoria / Empate / Derrota
    ├── Train: 5.011 partidos (2010–2025)
    └── Accuracy: 52% (vs 33% baseline aleatorio)
    │
    ▼
Simulación MCTS
    ├── 10.000 iteraciones por fase de grupos
    ├── Probabilidades de clasificación por grupo
    └── Cruces eliminatorios hasta la final
```

### Métricas del Modelo

| Métrica | Valor |
|---|---|
| Accuracy | **52%** |
| AUC — Victoria | **0.76** |
| AUC — Derrota | **0.77** |
| AUC — Empate | **0.58** |
| Partidos de entrenamiento | **5.011** |
| Período de datos | **2010 – 2025** |

---

## 📂 Estructura del Proyecto

```
SIM_MUNDIAL2026/
│
├── app.py                    # Aplicación principal Streamlit (1200+ líneas)
├── requirements.txt          # Dependencias del proyecto
├── df_mundial.csv            # Dataset de las 48 selecciones del Mundial 2026
├── df_train.csv              # Dataset de entrenamiento (5.011 partidos)
├── .gitignore
│
├── model/                    # Modelos y datos serializados
│   ├── modelo_bal.pkl        # Modelo XGBoost entrenado (balanced)
│   ├── datos_torneo.pkl      # Resultados de simulación del torneo
│   ├── elo_dict.pkl          # Diccionario de ratings ELO
│   ├── features_dict.pkl     # Features computados por selección
│   ├── grupos.pkl            # Configuración de grupos del Mundial
│   ├── resultados_grupos.pkl # Probabilidades de clasificación por grupo
│   └── tech_data.pkl         # Datos técnicos (ROC, Feature Importance, ELO)
│
├── content/                  # Datos fuente
│   ├── results.csv           # Resultados históricos de fútbol internacional (1872+)
│   └── national_teams.csv    # Datos de selecciones (Transfermarkt)
│
└── notebook/                 # Experimentación
    └── Mundial_2026_v2.ipynb # Notebook completo de desarrollo del modelo
```

---

## 🛠️ Stack Tecnológico

| Categoría | Tecnología | Uso |
|---|---|---|
| **ML Framework** | XGBoost 3.2 | Clasificador multiclase con pesos balanceados |
| **Data Processing** | Pandas 3.0, NumPy 2.4 | Manipulación de datos y feature engineering |
| **ML Utilities** | Scikit-learn 1.8 | Métricas, preprocesamiento, evaluación |
| **Visualization** | Plotly 6.7 | Gráficos interactivos (graph_objects) |
| **Web Framework** | Streamlit 1.57 | Dashboard interactivo con UI declarativa |
| **Serialization** | Pickle | Persistencia de modelos y datos precomputados |

---

## 🗺️ Roadmap

- [ ] Deploy en Streamlit Cloud / HuggingFace Spaces
- [ ] Añadir simulación en tiempo real con parámetros ajustables
- [ ] Integrar datos de apuestas como feature adicional
- [ ] Implementar backtesting con Qatar 2022
- [ ] Añadir capturas de pantalla y GIFs al README
- [ ] Exportar resultados a PDF interactivo
- [ ] API REST para consultar predicciones programáticamente

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 👨‍💻 Autor

Desarrollado con ☕ y datos por **vicentenr8**

---

<p align="center">
  <sub>Hecho con ❤️ para el Mundial 2026 🏆</sub>
</p>
