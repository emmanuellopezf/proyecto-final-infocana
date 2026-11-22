"""Dashboard de Indicadores y Pronóstico — Producción de Azúcar en México (Infocaña 2015-2026).

Framework: Shiny for Python (compatible con ShinyLive / GitHub Pages).
Modelo: Regresión Lineal supervisada (scikit-learn) con validación cronológica 80% / 20%.
Fuente de datos: CONADESUCA / Infocaña (datos.gob.mx).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Carga de datos procesados por el pipeline ETL (empaquetados en app/data/)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

DF_ZAFRA = pd.read_csv(DATA_DIR / "infocana_resumen_zafra.csv", encoding="utf-8")
DF_INGENIO = pd.read_csv(DATA_DIR / "infocana_ingenio_zafra.csv", encoding="utf-8")
DF_SEMANAL = pd.read_csv(DATA_DIR / "infocana_semanal_nacional.csv", encoding="utf-8")
ETL_AUDIT = json.loads((DATA_DIR / "etl_summary.json").read_text(encoding="utf-8"))

# Zafras completas para serie temporal anual (evita sesgo por zafra en curso 2025-2026)
DF_ZAFRA_COMPLETAS = DF_ZAFRA[DF_ZAFRA["estado_zafra"] == "Completa"].sort_values("anio_inicio").reset_index(drop=True)
DF_ZAFRA_PARCIAL = DF_ZAFRA[DF_ZAFRA["estado_zafra"] != "Completa"].reset_index(drop=True)

VAR_LABELS = {
    "azucar_producida_total": ("Azúcar producida total (ton)", "ton", 1_000_000.0, "M ton"),
    "cana_molida_neta": ("Caña molida neta (ton)", "ton", 1_000_000.0, "M ton"),
    "cana_molida_bruta": ("Caña molida bruta (ton)", "ton", 1_000_000.0, "M ton"),
    "superficie_cosechada": ("Superficie cosechada (ha)", "ha", 1_000.0, "mil ha"),
    "rendimiento_agroindustrial": ("Rendimiento agroindustrial (ton/ha)", "ton/ha", 1.0, "ton/ha"),
}

SEMANAL_COL_MAP = {
    "azucar_producida_total": ("azucar_acumulada_ton", "Azúcar acumulada (ton)", "ton", 1_000.0, "mil ton"),
    "cana_molida_neta": ("cana_neta_acumulada_ton", "Caña molida neta acumulada (ton)", "ton", 1_000.0, "mil ton"),
    "cana_molida_bruta": ("cana_neta_acumulada_ton", "Caña molida neta acumulada (ton)", "ton", 1_000.0, "mil ton"),
    "superficie_cosechada": ("superficie_acumulada_ha", "Superficie cosechada acumulada (ha)", "ha", 1_000.0, "mil ha"),
    "rendimiento_agroindustrial": ("azucar_acumulada_ton", "Azúcar acumulada (ton)", "ton", 1_000.0, "mil ton"),
}

X_OP_LABELS = {
    "cana_molida_neta": ("Caña molida neta (ton)", "ton"),
    "cana_molida_bruta": ("Caña molida bruta (ton)", "ton"),
    "superficie_cosechada": ("Superficie cosechada (ha)", "ha"),
}

ZAFRAS_LIST = ["Todas (2015-2026)", "Solo completas (2015-2025)"] + sorted(DF_INGENIO["zafra"].unique().tolist(), reverse=True)
INGENIOS_LIST = ["Todos los ingenios"] + sorted(DF_INGENIO["ingenio"].unique().tolist())


def fmt_num(val: float, decimals: int = 1) -> str:
    return f"{val:,.{decimals}f}"


def fmt_compact(val: float, divisor: float, short_unit: str) -> str:
    if divisor > 1.0:
        return f"{val / divisor:,.2f} {short_unit}"
    return f"{val:,.2f} {short_unit}"


# ---------------------------------------------------------------------------
# Estilos CSS personalizados (inspirados en el mockup del proyecto)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --navy-header: #1b365d;
  --navy-accent: #244b7a;
  --primary-blue: #1a66c4;
  --soft-bg: #eef2f7;
  --card-border: #d9e2ec;
  --kpi-blue-bg: #e8f1fb;
  --kpi-slate-bg: #eef3f8;
  --kpi-green-bg: #e6f5ec;
  --text-dark: #102a43;
  --text-muted: #486581;
}

body {
  background-color: var(--soft-bg);
  color: var(--text-dark);
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif;
}

.top-banner {
  background: linear-gradient(90deg, #162e4d 0%, #244b7a 100%);
  color: #ffffff;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(16, 42, 67, 0.18);
  margin-bottom: 14px;
}

.top-banner-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.top-banner h1 {
  font-size: 1.32rem;
  font-weight: 700;
  margin: 0;
  letter-spacing: 0.2px;
}

.top-banner-sub {
  font-size: 0.80rem;
  color: #cfe0f5;
  margin: 0;
}

.badge-unach {
  background: rgba(255, 255, 255, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.28);
  color: #f0f6fc;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
}

.nav-tabs {
  border-bottom: 2px solid #cbd5e1;
  margin-bottom: 16px;
}

.nav-tabs .nav-link {
  color: var(--text-muted);
  font-weight: 600;
  border: none;
  padding: 10px 18px;
}

.nav-tabs .nav-link.active {
  color: var(--navy-header);
  background-color: #ffffff;
  border-bottom: 3px solid var(--primary-blue);
  border-radius: 8px 8px 0 0;
}

.sidebar-panel {
  background: #ffffff;
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}

.sidebar-title {
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--navy-header);
  padding-bottom: 10px;
  margin-bottom: 14px;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-status-box {
  background: #e9f7ee;
  border: 1px solid #b7e4c7;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 14px;
  font-size: 0.86rem;
  color: #14532d;
}

.model-status-box .status-row {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 5px;
}

.model-status-box .metrics-mini {
  font-size: 0.80rem;
  color: #1e4620;
  background: rgba(255, 255, 255, 0.65);
  border-radius: 6px;
  padding: 7px 9px;
  margin-top: 6px;
  line-height: 1.45;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 16px;
}

@media (max-width: 900px) {
  .kpi-grid {
    grid-template-columns: 1fr;
  }
}

.kpi-card {
  border-radius: 12px;
  padding: 14px 18px;
  border: 1px solid var(--card-border);
  display: flex;
  align-items: center;
  gap: 14px;
  box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04);
}

.kpi-card.kpi-blue { background: linear-gradient(135deg, #e8f1fb 0%, #f5f9ff 100%); border-color: #c2daf5; }
.kpi-card.kpi-slate { background: linear-gradient(135deg, #edf2f7 0%, #f8fafc 100%); border-color: #cbd5e1; }
.kpi-card.kpi-green { background: linear-gradient(135deg, #e5f6ec 0%, #f4fbf7 100%); border-color: #b7e4c7; }

.kpi-icon {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(15, 23, 42, 0.08);
}

.kpi-label {
  font-size: 0.80rem;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 2px;
}

.kpi-value {
  font-size: 1.42rem;
  font-weight: 800;
  color: var(--text-dark);
  line-height: 1.15;
}

.kpi-card.kpi-green .kpi-value { color: #166534; }

.kpi-sub {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 3px;
}

.content-card {
  background: #ffffff;
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}

.content-card-title {
  font-size: 1.04rem;
  font-weight: 700;
  color: var(--navy-header);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.btn-generate {
  width: 100%;
  background: linear-gradient(180deg, #1d70d6 0%, #1454a8 100%);
  border: none;
  color: #ffffff;
  font-weight: 700;
  padding: 10px 14px;
  border-radius: 8px;
  margin-top: 6px;
  box-shadow: 0 2px 5px rgba(20, 84, 168, 0.25);
}

.btn-generate:hover {
  background: linear-gradient(180deg, #1660bd 0%, #10458c 100%);
  color: #ffffff;
}

.qa-card {
  background: #ffffff;
  border-left: 5px solid var(--primary-blue);
  border-top: 1px solid var(--card-border);
  border-right: 1px solid var(--card-border);
  border-bottom: 1px solid var(--card-border);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 16px;
}

.qa-card h4 {
  color: var(--navy-header);
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 10px;
}

.qa-highlight {
  background: #f0f7ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 10px 14px;
  margin: 10px 0;
  font-weight: 600;
  color: #1e3a8a;
}

.etl-pill-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}

@media (max-width: 900px) {
  .etl-pill-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.etl-pill {
  background: #ffffff;
  border: 1px solid var(--card-border);
  border-radius: 10px;
  padding: 12px 14px;
  text-align: center;
}

.etl-pill .num {
  font-size: 1.35rem;
  font-weight: 800;
  color: var(--navy-header);
}

.etl-pill .lbl {
  font-size: 0.78rem;
  color: var(--text-muted);
}
"""

SVG_CHART = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>"""
SVG_CAL = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a66c4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>"""
SVG_DB = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#334e68" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>"""
SVG_TREND = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#15803d" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>"""


# ---------------------------------------------------------------------------
# Interfaz de Usuario (UI)
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("Dashboard Infocaña — Pronóstico de Producción de Azúcar (CONADESUCA)"),
        ui.tags.style(CUSTOM_CSS),
    ),
    # Barra superior estilo mockup
    ui.div(
        ui.div(
            ui.HTML(SVG_CHART),
            ui.div(
                ui.h1("Dashboard - Pronóstico de Producción de Azúcar en México (Infocaña)"),
                ui.p(
                    "Pipeline ETL (2015–2026) · Regresión Lineal (scikit-learn) · Datos Abiertos CONADESUCA",
                    class_="top-banner-sub",
                ),
            ),
            class_="top-banner-left",
        ),
        ui.span("UC: Big Data · UNACH LIDTS", class_="badge-unach"),
        class_="top-banner",
    ),
    ui.div(
        ui.navset_tab(
            # ===============================================================
            # TAB 1: Pronóstico de Serie de Tiempo (Mockup principal)
            # ===============================================================
            ui.nav_panel(
                "📈 Serie de Tiempo y Pronóstico",
                ui.row(
                    ui.column(
                        3,
                        ui.div(
                            ui.div("⚙ Configuración del pronóstico", class_="sidebar-title"),
                            ui.input_select(
                                "ts_variable",
                                "Variable a pronosticar",
                                choices={k: v[0] for k, v in VAR_LABELS.items()},
                                selected="azucar_producida_total",
                            ),
                            ui.input_select(
                                "ts_frecuencia",
                                "Frecuencia / Serie",
                                choices={
                                    "anual": "Anual (Zafras completas 2015–2025)",
                                    "semanal_2526": "Semanal (Avance Zafra 2025–2026)",
                                    "semanal_2425": "Semanal (Zafra Completa 2024–2025)",
                                },
                                selected="anual",
                            ),
                            ui.input_radio_buttons(
                                "ts_horizonte",
                                "Horizonte de pronóstico",
                                choices={"1": "1 periodo", "2": "2 periodos", "3": "3 periodos"},
                                selected="1",
                                inline=True,
                            ),
                            ui.input_slider(
                                "ts_train_pct",
                                "División Entrenamiento (%) / Prueba",
                                min=60,
                                max=90,
                                value=80,
                                step=5,
                                post="%",
                            ),
                            ui.input_action_button(
                                "btn_generar",
                                "▶ Generar pronóstico",
                                class_="btn-generate",
                            ),
                            ui.output_ui("ts_model_status"),
                            class_="sidebar-panel",
                        ),
                    ),
                    ui.column(
                        9,
                        ui.output_ui("ts_kpis"),
                        ui.div(
                            ui.div(
                                ui.span("📉 Serie histórica y pronóstico (Regresión Lineal)"),
                                ui.span(
                                    "Validación cronológica 80% / 20%",
                                    style="font-size:0.80rem; color:#486581; font-weight:500;",
                                ),
                                class_="content-card-title",
                            ),
                            ui.output_plot("ts_plot", height="410px"),
                            class_="content-card",
                        ),
                        ui.div(
                            ui.div(
                                ui.span("📋 Tabla de datos históricos, evaluación y pronóstico (DataTable)"),
                                class_="content-card-title",
                            ),
                            ui.output_data_frame("ts_table"),
                            class_="content-card",
                        ),
                    ),
                ),
            ),
            # ===============================================================
            # TAB 2: Regresión Lineal Operativa Y = B0 + B1 * X
            # ===============================================================
            ui.nav_panel(
                "🧮 Regresión Operativa (Y = β₀ + β₁X)",
                ui.row(
                    ui.column(
                        3,
                        ui.div(
                            ui.div("🔬 Configuración del modelo operativo", class_="sidebar-title"),
                            ui.input_select(
                                "op_x_var",
                                "Variable independiente (X)",
                                choices={k: v[0] for k, v in X_OP_LABELS.items()},
                                selected="cana_molida_neta",
                            ),
                            ui.input_select(
                                "op_zafra_scope",
                                "Alcance de zafras",
                                choices=ZAFRAS_LIST,
                                selected="Todas (2015-2026)",
                            ),
                            ui.input_slider(
                                "op_train_pct",
                                "Entrenamiento (%) / Prueba",
                                min=60,
                                max=90,
                                value=80,
                                step=5,
                                post="%",
                            ),
                            ui.hr(),
                            ui.tags.label("Simulador: ingresa valor de X (por ingenio)", class_="fw-bold mb-1"),
                            ui.input_numeric(
                                "op_sim_x",
                                "Valor de X a simular",
                                value=1_000_000,
                                min=1_000,
                                max=5_000_000,
                                step=50_000,
                            ),
                            ui.output_ui("op_model_status"),
                            class_="sidebar-panel",
                        ),
                    ),
                    ui.column(
                        9,
                        ui.output_ui("op_kpis"),
                        ui.div(
                            ui.div(
                                ui.span("📊 Dispersión y Recta de Regresión Lineal Simple (sklearn)"),
                                ui.span(
                                    "Variable dependiente Y: azucar_producida_total (ton)",
                                    style="font-size:0.80rem; color:#486581; font-weight:500;",
                                ),
                                class_="content-card-title",
                            ),
                            ui.output_plot("op_plot", height="400px"),
                            class_="content-card",
                        ),
                        ui.div(
                            ui.div(
                                ui.span("📋 Resultados en conjunto de prueba (20%) — Real vs Predicho"),
                                class_="content-card-title",
                            ),
                            ui.output_data_frame("op_table"),
                            class_="content-card",
                        ),
                    ),
                ),
            ),
            # ===============================================================
            # TAB 3: Explorador ETL y Datos por Ingenio
            # ===============================================================
            ui.nav_panel(
                "🗄 Explorador ETL e Ingenios",
                ui.output_ui("etl_kpi_strip"),
                ui.row(
                    ui.column(
                        4,
                        ui.div(
                            ui.input_select("exp_zafra", "Filtrar por Zafra", choices=ZAFRAS_LIST, selected="2024-2025"),
                            ui.input_select("exp_ingenio", "Filtrar por Ingenio", choices=INGENIOS_LIST, selected="Todos los ingenios"),
                            ui.p(
                                "Nota ETL: Los reportes de Infocaña son acumulados semanales. Para evitar doble conteo, "
                                "el pipeline extrae la semana de cierre máxima por (zafra, ingenio) e imputa caña molida neta "
                                f"faltante mediante la mediana histórica neta/bruta ({ETL_AUDIT['ratio_mediana_neta_bruta']:.3f}).",
                                style="font-size:0.82rem; color:#486581; margin-top:10px;",
                            ),
                            class_="sidebar-panel",
                        ),
                    ),
                    ui.column(
                        8,
                        ui.div(
                            ui.div("🏭 Top 10 Ingenios por Producción Total de Azúcar", class_="content-card-title"),
                            ui.output_plot("exp_top_plot", height="320px"),
                            class_="content-card",
                        ),
                    ),
                ),
                ui.div(
                    ui.div("📋 Conjunto de Datos Procesado (DataTable interactivo)", class_="content-card-title"),
                    ui.output_data_frame("exp_table"),
                    class_="content-card",
                ),
            ),
            # ===============================================================
            # TAB 4: Interpretación para Toma de Decisiones (Paso 3)
            # ===============================================================
            ui.nav_panel(
                "🧭 Interpretación y Decisiones",
                ui.output_ui("interpretation_panel"),
            ),
        ),
        style="max-width: 1440px; margin: 0 auto; padding: 0 18px 28px 18px;",
    ),
)


# ---------------------------------------------------------------------------
# Lógica Reactiva y Modelado (Server)
# ---------------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session) -> None:
    # Actualizar opciones de horizonte y valor sugerido al cambiar frecuencia
    @reactive.effect
    def _sync_horizon_choices() -> None:
        freq = input.ts_frecuencia()
        if freq == "anual":
            ui.update_radio_buttons(
                "ts_horizonte",
                choices={"1": "1 zafra", "2": "2 zafras", "3": "3 zafras"},
                selected="3",
                inline=True,
            )
        else:
            ui.update_radio_buttons(
                "ts_horizonte",
                choices={"6": "6 sem.", "12": "12 sem.", "24": "24 sem."},
                selected="12",
                inline=True,
            )

    @reactive.effect
    def _sync_sim_x_default() -> None:
        x_var = input.op_x_var()
        if x_var == "superficie_cosechada":
            ui.update_numeric("op_sim_x", value=15_000, min=100, max=100_000, step=1_000)
        else:
            ui.update_numeric("op_sim_x", value=1_000_000, min=10_000, max=5_000_000, step=50_000)

    # -----------------------------------------------------------------------
    # Modelo 1: Serie de tiempo con Regresión Lineal (sklearn)
    # -----------------------------------------------------------------------
    @reactive.calc
    def ts_results() -> dict:
        # Dependencia explícita del botón "Generar pronóstico" y controles
        _ = input.btn_generar()
        var = input.ts_variable()
        freq = input.ts_frecuencia()
        horizon = int(input.ts_horizonte() or 1)
        train_pct = int(input.ts_train_pct()) / 100.0

        if freq == "anual":
            df = DF_ZAFRA_COMPLETAS.copy()
            label_title, unit, divisor, short_unit = VAR_LABELS[var]
            x_vals = np.arange(1, len(df) + 1, dtype=float)
            y_vals = df[var].astype(float).values
            labels = df["zafra"].tolist()
            last_start = int(df["anio_inicio"].iloc[-1])
            fut_x = np.arange(len(df) + 1, len(df) + horizon + 1, dtype=float)
            fut_labels = [f"{last_start + i}-{last_start + i + 1}*" for i in range(1, horizon + 1)]
            horizon_text = f"{horizon} {'zafra' if horizon == 1 else 'zafras'}"
            x_axis_label = "Zafra"
            last_period_note = f"Última zafra completa: {labels[-1]}"
        else:
            zafra_target = "2025-2026" if freq == "semanal_2526" else "2024-2025"
            col_sem, label_title, unit, divisor, short_unit = SEMANAL_COL_MAP[var]
            df = DF_SEMANAL[DF_SEMANAL["zafra"] == zafra_target].sort_values("no_s_zafra").reset_index(drop=True)
            x_vals = df["no_s_zafra"].astype(float).values
            y_vals = df[col_sem].astype(float).values
            labels = [f"S{int(s)}" for s in df["no_s_zafra"]]
            last_sem = int(df["no_s_zafra"].iloc[-1])
            fut_x = np.arange(last_sem + 1, last_sem + horizon + 1, dtype=float)
            fut_labels = [f"S{int(s)}*" for s in fut_x]
            horizon_text = f"{horizon} semanas"
            x_axis_label = f"Semana de Zafra ({zafra_target})"
            last_period_note = f"Zafra {zafra_target} · Semana {last_sem}"

        n = len(x_vals)
        n_train = max(2, min(n - 1, int(round(n * train_pct))))
        X_all = x_vals.reshape(-1, 1)
        X_train, y_train = X_all[:n_train], y_vals[:n_train]
        X_test, y_test = X_all[n_train:], y_vals[n_train:]

        # Ajuste en entrenamiento (80% por defecto) para evaluación rigurosa sin fuga temporal
        model_eval = LinearRegression().fit(X_train, y_train)
        pred_train = model_eval.predict(X_train)
        pred_test = model_eval.predict(X_test)

        r2_train = float(r2_score(y_train, pred_train))
        r2_test = float(r2_score(y_test, pred_test)) if len(y_test) >= 4 else float("nan")
        rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))
        mae_test = float(mean_absolute_error(y_test, pred_test))
        mape_test = float(np.mean(np.abs((y_test - pred_test) / np.maximum(np.abs(y_test), 1e-9))) * 100.0)

        # Modelo calibrado con toda la serie histórica para proyectar hacia el futuro
        model_full = LinearRegression().fit(X_all, y_vals)
        pred_full_hist = model_full.predict(X_all)
        rmse_full = float(np.sqrt(mean_squared_error(y_vals, pred_full_hist)))
        r2_full = float(r2_score(y_vals, pred_full_hist))
        fut_pred = np.clip(model_full.predict(fut_x.reshape(-1, 1)), a_min=0.0, a_max=None)

        # Construir DataFrame resumen para DataTable
        hist_rows = []
        for i in range(n):
            part = "Entrenamiento (80%)" if i < n_train and int(input.ts_train_pct()) == 80 else (
                f"Entrenamiento ({input.ts_train_pct()}%)" if i < n_train else f"Prueba ({100 - int(input.ts_train_pct())}%)"
            )
            est_eval = float(pred_train[i]) if i < n_train else float(pred_test[i - n_train])
            hist_rows.append(
                {
                    "Periodo": labels[i],
                    "Tipo": part,
                    "Valor Real": round(float(y_vals[i]), 2),
                    "Ajuste Lineal": round(float(pred_full_hist[i]), 2),
                    "Predicción Split": round(est_eval, 2),
                    "Error Abs.": round(abs(float(y_vals[i]) - est_eval), 2),
                }
            )
        for j in range(horizon):
            hist_rows.append(
                {
                    "Periodo": fut_labels[j],
                    "Tipo": "Pronóstico",
                    "Valor Real": np.nan,
                    "Ajuste Lineal": round(float(fut_pred[j]), 2),
                    "Predicción Split": round(float(fut_pred[j]), 2),
                    "Error Abs.": np.nan,
                }
            )

        return {
            "var": var,
            "freq": freq,
            "label_title": label_title,
            "unit": unit,
            "divisor": divisor,
            "short_unit": short_unit,
            "horizon": horizon,
            "horizon_text": horizon_text,
            "x_axis_label": x_axis_label,
            "last_period_note": last_period_note,
            "x_vals": x_vals,
            "y_vals": y_vals,
            "labels": labels,
            "n_train": n_train,
            "n_test": n - n_train,
            "fut_x": fut_x,
            "fut_labels": fut_labels,
            "fut_pred": fut_pred,
            "pred_full_hist": pred_full_hist,
            "pred_eval_all": np.concatenate([pred_train, pred_test]),
            "beta0": float(model_full.intercept_),
            "beta1": float(model_full.coef_[0]),
            "r2_train": r2_train,
            "r2_test": r2_test,
            "r2_full": r2_full,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "mape_test": mape_test,
            "rmse_full": rmse_full,
            "table_df": pd.DataFrame(hist_rows),
        }

    @render.ui
    def ts_model_status():
        res = ts_results()
        sign = "+" if res["beta1"] >= 0 else "-"
        extra_metric = (
            f"<strong>R² (Prueba):</strong> {res['r2_test']:.3f}"
            if not np.isnan(res["r2_test"])
            else f"<strong>MAPE (Prueba):</strong> {res['mape_test']:.1f}%"
        )
        return ui.div(
            ui.div(ui.HTML("<strong>📊 Modelo:</strong> Regresión Lineal (sklearn)"), class_="status-row"),
            ui.div(ui.HTML("<strong>✅ Estado:</strong> Pronóstico generado"), class_="status-row"),
            ui.div(
                ui.HTML(
                    f"<strong>Ecuación:</strong> Y = {res['beta0']:,.1f} {sign} {abs(res['beta1']):,.1f}·t<br>"
                    f"<strong>Split:</strong> {res['n_train']} train / {res['n_test']} test<br>"
                    f"<strong>R² Tendencia:</strong> {res['r2_full']:.3f}<br>"
                    f"<strong>RMSE (Prueba):</strong> {fmt_compact(res['rmse_test'], res['divisor'], res['short_unit'])}<br>"
                    f"<strong>MAE (Prueba):</strong> {fmt_compact(res['mae_test'], res['divisor'], res['short_unit'])}<br>"
                    f"{extra_metric}"
                ),
                class_="metrics-mini",
            ),
            class_="model-status-box",
        )

    @render.ui
    def ts_kpis():
        res = ts_results()
        last_val = float(res["y_vals"][-1])
        final_pred = float(res["fut_pred"][-1])
        delta_pct = ((final_pred - last_val) / last_val * 100.0) if last_val else 0.0
        delta_sign = "+" if delta_pct >= 0 else ""

        extra_sub = res["last_period_note"]
        if res["freq"] == "anual" and res["var"] == "azucar_producida_total" and len(DF_ZAFRA_PARCIAL):
            parcial_val = float(DF_ZAFRA_PARCIAL["azucar_producida_total"].iloc[0])
            sem_max = int(DF_ZAFRA_PARCIAL["semana_cierre_max"].iloc[0])
            extra_sub = f"Cierre 2024–2025 · Avance 25–26 (S{sem_max}): {parcial_val/1e6:.2f} M ton"

        return ui.div(
            ui.div(
                ui.div(ui.HTML(SVG_CAL), class_="kpi-icon"),
                ui.div(
                    ui.div("Horizonte:", class_="kpi-label"),
                    ui.div(res["horizon_text"], class_="kpi-value"),
                    ui.div(f"Hasta {res['fut_labels'][-1].replace('*', '')}", class_="kpi-sub"),
                ),
                class_="kpi-card kpi-blue",
            ),
            ui.div(
                ui.div(ui.HTML(SVG_DB), class_="kpi-icon"),
                ui.div(
                    ui.div("Último valor real:", class_="kpi-label"),
                    ui.div(fmt_compact(last_val, res["divisor"], res["short_unit"]), class_="kpi-value"),
                    ui.div(extra_sub, class_="kpi-sub"),
                ),
                class_="kpi-card kpi-slate",
            ),
            ui.div(
                ui.div(ui.HTML(SVG_TREND), class_="kpi-icon"),
                ui.div(
                    ui.div("Pronóstico final:", class_="kpi-label"),
                    ui.div(fmt_compact(final_pred, res["divisor"], res["short_unit"]), class_="kpi-value"),
                    ui.div(f"Variación estimada: {delta_sign}{delta_pct:.1f}%", class_="kpi-sub"),
                ),
                class_="kpi-card kpi-green",
            ),
            class_="kpi-grid",
        )

    @render.plot
    def ts_plot():
        res = ts_results()
        div = res["divisor"]
        short_u = res["short_unit"]

        y_hist = res["y_vals"] / div
        y_fit = res["pred_full_hist"] / div
        y_fut = res["fut_pred"] / div
        band = res["rmse_full"] / div

        all_labels = res["labels"] + res["fut_labels"]
        idx_hist = np.arange(len(y_hist))
        idx_fut = np.arange(len(y_hist), len(all_labels))

        fig, ax = plt.subplots(figsize=(10.2, 4.3), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafcff")

        # Sombreado zona de prueba (20%)
        n_train = res["n_train"]
        ax.axvspan(n_train - 0.5, len(y_hist) - 1, color="#fff7ed", alpha=0.85, zorder=1, label=f"Zona prueba ({100 - int(input.ts_train_pct())}%)")

        # Línea histórica (azul estilo mockup)
        ax.plot(idx_hist, y_hist, color="#1d6fd8", marker="o", linewidth=2.4, markersize=6.5, label="Datos históricos", zorder=4)

        # Línea de tendencia lineal histórica (punteada sutil)
        ax.plot(idx_hist, y_fit, color="#64748b", linestyle="--", linewidth=1.4, alpha=0.85, label="Tendencia lineal (OLS)", zorder=3)

        # Conector + pronóstico (verde estilo mockup)
        idx_conn = np.concatenate([[idx_hist[-1]], idx_fut])
        y_conn = np.concatenate([[y_hist[-1]], y_fut])
        ax.plot(idx_conn, y_conn, color="#16a34a", marker="o", linewidth=2.5, markersize=7, label="Pronóstico", zorder=5)
        ax.fill_between(idx_fut, np.maximum(0, y_fut - band), y_fut + band, color="#22c55e", alpha=0.16, label="Banda ±1 RMSE", zorder=2)

        # Línea vertical de corte + caja "Último dato" (como el mockup)
        ax.axvline(idx_hist[-1], color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
        last_y_disp = y_hist[-1]
        y_min_all = min(float(np.min(y_hist)), float(np.min(y_fut)))
        y_max_all = max(float(np.max(y_hist)), float(np.max(y_fut)))
        y_span = max(y_max_all - y_min_all, 0.5)
        ax.annotate(
            f"Último dato: {last_y_disp:,.2f} {short_u}",
            xy=(idx_hist[-1], last_y_disp),
            xytext=(max(0, idx_hist[-1] - 2.6), min(y_max_all + y_span * 0.16, last_y_disp + y_span * 0.52)),
            fontsize=8.8,
            fontweight="bold",
            color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#e8f1fb", edgecolor="#93c5fd", linewidth=1.1),
            arrowprops=dict(arrowstyle="->", color="#1d6fd8", lw=1.2),
            zorder=6,
        )

        # Etiqueta del pronóstico final
        ax.annotate(
            f"{y_fut[-1]:,.2f} {short_u}",
            xy=(idx_fut[-1], y_fut[-1]),
            xytext=(idx_fut[-1] - 0.35, y_fut[-1] + y_span * 0.10),
            fontsize=8.6,
            fontweight="bold",
            color="#14532d",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#dcfce7", edgecolor="#86efac", linewidth=1.0),
            zorder=6,
        )

        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=28 if len(all_labels) > 9 else 15, ha="right", fontsize=8.5)
        ax.set_ylabel(f"{res['label_title']} ({short_u})", fontsize=9.5, fontweight="bold", color="#102a43")
        ax.set_xlabel(res["x_axis_label"], fontsize=9.5, fontweight="bold", color="#102a43", labelpad=6)
        ax.grid(True, linestyle=":", alpha=0.55, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(max(0, y_min_all - y_span * 0.22), y_max_all + y_span * 0.30)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=8.3)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, bottom=0.22)
        return fig

    @render.data_frame
    def ts_table():
        return render.DataTable(ts_results()["table_df"], filters=False, summary=True, height="280px")

    # -----------------------------------------------------------------------
    # Modelo 2: Regresión Lineal Operativa Y = B0 + B1 * X (por ingenio/zafra)
    # -----------------------------------------------------------------------
    @reactive.calc
    def op_results() -> dict:
        x_var = input.op_x_var()
        scope = input.op_zafra_scope()
        train_pct = int(input.op_train_pct()) / 100.0
        sim_x = float(input.op_sim_x() or 100_000)

        df = DF_INGENIO.copy()
        if scope == "Solo completas (2015-2025)":
            df = df[df["estado_zafra"] == "Completa"].copy()
        elif scope != "Todas (2015-2026)":
            df = df[df["zafra"] == scope].copy()

        df = df[(df[x_var] > 0) & (df["azucar_producida_total"] > 0)].sort_values(["anio_inicio", "ingenio"]).reset_index(drop=True)

        # División cronológica 80/20 (sin fuga de datos futuros)
        n = len(df)
        n_train = max(3, min(n - 2, int(round(n * train_pct))))
        df_train = df.iloc[:n_train].copy()
        df_test = df.iloc[n_train:].copy()

        X_train = df_train[[x_var]].values
        y_train = df_train["azucar_producida_total"].values
        X_test = df_test[[x_var]].values
        y_test = df_test["azucar_producida_total"].values

        model = LinearRegression().fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        sim_y = max(0.0, float(model.predict([[sim_x]])[0]))

        r2_tr = float(r2_score(y_train, pred_train))
        r2_te = float(r2_score(y_test, pred_test))
        rmse_te = float(np.sqrt(mean_squared_error(y_test, pred_test)))
        mae_te = float(mean_absolute_error(y_test, pred_test))

        test_table = pd.DataFrame(
            {
                "Zafra": df_test["zafra"].values,
                "Ingenio": df_test["ingenio"].values,
                X_OP_LABELS[x_var][0]: np.round(df_test[x_var].values, 1),
                "Azúcar Real (ton)": np.round(y_test, 1),
                "Azúcar Predicho (ton)": np.round(pred_test, 1),
                "Error Abs. (ton)": np.round(np.abs(y_test - pred_test), 1),
            }
        )

        return {
            "x_var": x_var,
            "x_label": X_OP_LABELS[x_var][0],
            "x_unit": X_OP_LABELS[x_var][1],
            "df_train": df_train,
            "df_test": df_test,
            "beta0": float(model.intercept_),
            "beta1": float(model.coef_[0]),
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "rmse_test": rmse_te,
            "mae_test": mae_te,
            "sim_x": sim_x,
            "sim_y": sim_y,
            "test_table": test_table,
            "n_total": n,
        }

    @render.ui
    def op_model_status():
        res = op_results()
        sign = "+" if res["beta1"] >= 0 else "-"
        return ui.div(
            ui.div(ui.HTML("<strong>📊 Modelo:</strong> OLS LinearRegression"), class_="status-row"),
            ui.div(ui.HTML(f"<strong>✅ Registros:</strong> {res['n_total']} ingenio-zafra"), class_="status-row"),
            ui.div(
                ui.HTML(
                    f"<strong>Ecuación:</strong><br>Y = {res['beta0']:,.1f} {sign} {abs(res['beta1']):,.4f}·X<br>"
                    f"<strong>R² Entrenamiento:</strong> {res['r2_train']:.4f}<br>"
                    f"<strong>R² Prueba:</strong> {res['r2_test']:.4f}<br>"
                    f"<strong>RMSE Prueba:</strong> {res['rmse_test']:,.1f} ton<br>"
                    f"<strong>MAE Prueba:</strong> {res['mae_test']:,.1f} ton"
                ),
                class_="metrics-mini",
            ),
            class_="model-status-box",
        )

    @render.ui
    def op_kpis():
        res = op_results()
        sign = "+" if res["beta1"] >= 0 else "-"
        return ui.div(
            ui.div(
                ui.div(ui.HTML(SVG_CAL), class_="kpi-icon"),
                ui.div(
                    ui.div("Pendiente (β₁):", class_="kpi-label"),
                    ui.div(f"{res['beta1']:.4f}", class_="kpi-value"),
                    ui.div(f"ton azúcar por {res['x_unit']} de X", class_="kpi-sub"),
                ),
                class_="kpi-card kpi-blue",
            ),
            ui.div(
                ui.div(ui.HTML(SVG_DB), class_="kpi-icon"),
                ui.div(
                    ui.div("Precisión R² (Prueba):", class_="kpi-label"),
                    ui.div(f"{res['r2_test']*100:.2f}%", class_="kpi-value"),
                    ui.div(f"RMSE: {res['rmse_test']:,.0f} ton · MAE: {res['mae_test']:,.0f} ton", class_="kpi-sub"),
                ),
                class_="kpi-card kpi-slate",
            ),
            ui.div(
                ui.div(ui.HTML(SVG_TREND), class_="kpi-icon"),
                ui.div(
                    ui.div("Pronóstico Simulado:", class_="kpi-label"),
                    ui.div(f"{res['sim_y']:,.0f} ton", class_="kpi-value"),
                    ui.div(f"Con X = {res['sim_x']:,.0f} {res['x_unit']} ({sign})", class_="kpi-sub"),
                ),
                class_="kpi-card kpi-green",
            ),
            class_="kpi-grid",
        )

    @render.plot
    def op_plot():
        res = op_results()
        x_var = res["x_var"]
        tr = res["df_train"]
        te = res["df_test"]

        fig, ax = plt.subplots(figsize=(10.0, 4.1), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafcff")

        ax.scatter(tr[x_var], tr["azucar_producida_total"], color="#2563eb", alpha=0.55, s=36, label=f"Entrenamiento ({len(tr)})")
        ax.scatter(te[x_var], te["azucar_producida_total"], color="#f59e0b", alpha=0.80, s=42, edgecolors="#92400e", linewidths=0.5, label=f"Prueba ({len(te)})")

        x_max = max(float(tr[x_var].max()), float(te[x_var].max()), res["sim_x"])
        x_line = np.linspace(0, x_max * 1.03, 120)
        y_line = np.maximum(0, res["beta0"] + res["beta1"] * x_line)
        ax.plot(x_line, y_line, color="#15803d", linewidth=2.4, label=f"Regresión lineal (R²={res['r2_test']:.3f})")

        ax.scatter([res["sim_x"]], [res["sim_y"]], color="#dc2626", marker="*", s=220, zorder=6, label=f"Simulación ({res['sim_y']:,.0f} ton)")

        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel(res["x_label"], fontsize=9.5, fontweight="bold", color="#102a43")
        ax.set_ylabel("Azúcar producida total (ton)", fontsize=9.5, fontweight="bold", color="#102a43")
        ax.grid(True, linestyle=":", alpha=0.55, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.5)
        fig.tight_layout()
        return fig

    @render.data_frame
    def op_table():
        return render.DataTable(op_results()["test_table"], filters=True, summary=True, height="280px")

    # -----------------------------------------------------------------------
    # Tab 3: Explorador ETL y Datos por Ingenio
    # -----------------------------------------------------------------------
    @render.ui
    def etl_kpi_strip():
        return ui.div(
            ui.div(ui.div(f"{ETL_AUDIT['archivos_procesados']}", class_="num"), ui.div("Datasets CSV integrados (2015–2026)", class_="lbl"), class_="etl-pill"),
            ui.div(ui.div(f"{ETL_AUDIT['registros_crudos_totales']:,}", class_="num"), ui.div("Registros crudos procesados", class_="lbl"), class_="etl-pill"),
            ui.div(ui.div(f"{ETL_AUDIT['registros_cierre_ingenio_zafra']:,}", class_="num"), ui.div("Cierres limpios (Ingenio × Zafra)", class_="lbl"), class_="etl-pill"),
            ui.div(ui.div(f"{ETL_AUDIT['ingenios_unicos']}", class_="num"), ui.div("Ingenios azucareros normalizados", class_="lbl"), class_="etl-pill"),
            class_="etl-pill-grid",
        )

    @reactive.calc
    def exp_filtered() -> pd.DataFrame:
        df = DF_INGENIO.copy()
        z = input.exp_zafra()
        ing = input.exp_ingenio()
        if z == "Solo completas (2015-2025)":
            df = df[df["estado_zafra"] == "Completa"]
        elif z != "Todas (2015-2026)":
            df = df[df["zafra"] == z]
        if ing != "Todos los ingenios":
            df = df[df["ingenio"] == ing]
        return df.sort_values(["zafra", "azucar_producida_total"], ascending=[False, False])

    @render.plot
    def exp_top_plot():
        df = exp_filtered()
        top = (
            df.groupby("ingenio", as_index=False)["azucar_producida_total"]
            .sum()
            .sort_values("azucar_producida_total", ascending=True)
            .tail(10)
        )
        fig, ax = plt.subplots(figsize=(8.0, 3.3), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafcff")
        bars = ax.barh(top["ingenio"], top["azucar_producida_total"] / 1_000.0, color="#1d6fd8", edgecolor="#164e9e")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(top["azucar_producida_total"].max() / 1_000.0 * 0.012, 0.5), bar.get_y() + bar.get_height() / 2, f"{w:,.1f}k", va="center", fontsize=8.0, color="#102a43")
        ax.set_xlabel("Azúcar producida total (miles de toneladas)", fontsize=9.0, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_xlim(0, max(top["azucar_producida_total"].max() / 1_000.0 * 1.18, 1.0))
        fig.tight_layout()
        return fig

    @render.data_frame
    def exp_table():
        cols = [
            "zafra",
            "estado_zafra",
            "ingenio",
            "no_s_zafra",
            "cana_molida_neta",
            "superficie_cosechada",
            "azucar_producida_total",
            "rendimiento_campo",
            "rendimiento_fabrica",
            "rendimiento_agroindustrial",
            "rendimiento_promedio_ingenio",
        ]
        df_show = exp_filtered()[cols].rename(
            columns={
                "zafra": "Zafra",
                "estado_zafra": "Estado",
                "ingenio": "Ingenio",
                "no_s_zafra": "Sem. cierre",
                "cana_molida_neta": "Caña Neta (ton)",
                "superficie_cosechada": "Superficie (ha)",
                "azucar_producida_total": "Azúcar Total (ton)",
                "rendimiento_campo": "Rend. Campo",
                "rendimiento_fabrica": "Rend. Fábrica (%)",
                "rendimiento_agroindustrial": "Rend. Agroind.",
                "rendimiento_promedio_ingenio": "Rend. Prom. Hist.",
            }
        )
        return render.DataTable(df_show, filters=True, summary=True, height="340px")

    # -----------------------------------------------------------------------
    # Tab 4: Interpretación de Resultados (Paso 3 del Proyecto)
    # -----------------------------------------------------------------------
    @render.ui
    def interpretation_panel():
        # Cálculos exactos sobre la serie anual completa (2015-2025)
        df_c = DF_ZAFRA_COMPLETAS
        x_idx = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        y_az = df_c["azucar_producida_total"].values
        lr_time = LinearRegression().fit(x_idx, y_az)
        pred_next_1 = float(lr_time.predict([[len(df_c) + 1]])[0])  # Zafra 2025-2026 cierre estimado
        pred_next_2 = float(lr_time.predict([[len(df_c) + 2]])[0])  # Zafra 2026-2027 estimado
        slope_anual = float(lr_time.coef_[0])

        # Modelo operativo global (azucar ~ cana_molida_neta)
        df_op = DF_INGENIO[(DF_INGENIO["cana_molida_neta"] > 0) & (DF_INGENIO["azucar_producida_total"] > 0)]
        n_tr = int(round(len(df_op) * 0.80))
        lr_op = LinearRegression().fit(df_op[["cana_molida_neta"]].iloc[:n_tr], df_op["azucar_producida_total"].iloc[:n_tr])
        r2_op_test = float(r2_score(df_op["azucar_producida_total"].iloc[n_tr:], lr_op.predict(df_op[["cana_molida_neta"]].iloc[n_tr:])))

        last_complete_zafra = df_c["zafra"].iloc[-1]
        last_complete_val = float(df_c["azucar_producida_total"].iloc[-1])
        first_zafra = df_c["zafra"].iloc[0]
        first_val = float(df_c["azucar_producida_total"].iloc[0])
        max_row = df_c.loc[df_c["azucar_producida_total"].idxmax()]
        min_row = df_c.loc[df_c["azucar_producida_total"].idxmin()]
        parcial_val = float(DF_ZAFRA_PARCIAL["azucar_producida_total"].iloc[0]) if len(DF_ZAFRA_PARCIAL) else 0.0
        parcial_sem = int(DF_ZAFRA_PARCIAL["semana_cierre_max"].iloc[0]) if len(DF_ZAFRA_PARCIAL) else 0
        cambio_total_pct = (last_complete_val - first_val) / first_val * 100.0

        return ui.div(
            ui.div(
                ui.h4("1. ¿Cuál es el último valor real de los datos históricos?"),
                ui.div(
                    f"Última zafra anual completa ({last_complete_zafra}): {last_complete_val:,.1f} toneladas "
                    f"({last_complete_val / 1e6:.2f} millones de toneladas) en {int(df_c['num_ingenios'].iloc[-1])} ingenios activos.",
                    class_="qa-highlight",
                ),
                ui.p(
                    f"Adicionalmente, el archivo más reciente del portal datos.gob.mx corresponde a la zafra en curso "
                    f"2025-2026 con corte preliminar en la semana {parcial_sem}, registrando un avance acumulado de "
                    f"{parcial_val:,.2f} toneladas en {int(DF_ZAFRA_PARCIAL['num_ingenios'].iloc[0])} ingenios. "
                    f"Durante la etapa ETL este ciclo parcial se separa explícitamente de la serie anual cerrada para evitar "
                    f"sesgar la pendiente de regresión anual."
                ),
                class_="qa-card",
            ),
            ui.div(
                ui.h4("2. ¿Cómo ha cambiado la producción total de azúcar a lo largo de las zafras y cuánto podría producirse en la próxima zafra (anual) si la tendencia continúa?"),
                ui.div(
                    f"Tendencia histórica ({first_zafra} a {last_complete_zafra}): descenso promedio de "
                    f"{abs(slope_anual):,.0f} toneladas por zafra ({cambio_total_pct:.1f}% acumulado). "
                    f"Pronóstico lineal cierre 2025-2026: {pred_next_1:,.0f} ton ({pred_next_1/1e6:.2f} M ton) · "
                    f"Pronóstico 2026-2027: {pred_next_2:,.0f} ton ({pred_next_2/1e6:.2f} M ton).",
                    class_="qa-highlight",
                ),
                ui.p(
                    f"Entre {first_zafra} ({first_val/1e6:.2f} M ton) y {last_complete_zafra} ({last_complete_val/1e6:.2f} M ton), "
                    f"la producción nacional alcanzó su punto máximo en la zafra {max_row['zafra']} con "
                    f"{max_row['azucar_producida_total']:,.1f} toneladas ({max_row['azucar_producida_total']/1e6:.2f} M ton) "
                    f"y su mínimo reciente en {min_row['zafra']} con {min_row['azucar_producida_total']:,.1f} toneladas "
                    f"({min_row['azucar_producida_total']/1e6:.2f} M ton), reflejando afectaciones climáticas (sequías 2019-2020 y 2023-2024) "
                    f"y una ligera reducción en el número de ingenios operando (de 51 a 47)."
                ),
                ui.p(
                    f"Complementariamente, el modelo de regresión lineal operativa (Azúcar Total = β₀ + β₁ · Caña Molida Neta) "
                    f"evaluado con división cronológica 80%/20% alcanza un R² = {r2_op_test:.4f} en pruebas, demostrando que cada "
                    f"tonelada adicional de caña molida neta aporta aproximadamente {lr_op.coef_[0]*100:.2f}% ({lr_op.coef_[0]:.4f} ton) "
                    f"de azúcar producida. Esto permite a los tomadores de decisiones planear capacidad fabril, inventarios y abasto nacional."
                ),
                class_="qa-card",
            ),
            ui.div(
                ui.h4("3. Recomendaciones Estratégicas para la Toma de Decisiones (CONADESUCA / Ingenios)"),
                ui.tags.ul(
                    ui.tags.li("Planificación de abasto nacional: Considerar la tendencia descendente (~153 mil ton/zafra) para anticipar balances de oferta y demanda antes de comprometer cupos de exportación."),
                    ui.tags.li("Control de rendimiento agroindustrial: Priorizar tecnificación de riego y renovación de variedades de caña en zonas afectadas por sequía para recuperar niveles superiores a 5.5 M ton."),
                    ui.tags.li("Monitoreo temprano con el modelo operativo: Dado el R² > 0.96 entre caña molida neta y azúcar producida, las proyecciones de molienda de campo permiten estimar con alta precisión el cierre de fábrica desde las primeras semanas de zafra."),
                ),
                class_="qa-card",
            ),
        )


app = App(app_ui, server)
