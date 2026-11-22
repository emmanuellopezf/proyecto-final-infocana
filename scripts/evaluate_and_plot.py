"""Evaluación estadística de los modelos de Regresión Lineal (scikit-learn)
y generación de gráficas de alta resolución para el informe técnico (.pdf).
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df_zafra = pd.read_csv(PROC_DIR / "infocana_resumen_zafra.csv", encoding="utf-8")
    df_ing = pd.read_csv(PROC_DIR / "infocana_ingenio_zafra.csv", encoding="utf-8")
    df_c = df_zafra[df_zafra["estado_zafra"] == "Completa"].sort_values("anio_inicio").reset_index(drop=True)
    df_p = df_zafra[df_zafra["estado_zafra"] != "Completa"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Serie Temporal Anual: azucar_producida_total ~ tiempo (zafra)
    #    Split cronológico 80% entrenamiento / 20% prueba
    # ------------------------------------------------------------------
    X_t = np.arange(1, len(df_c) + 1, dtype=float).reshape(-1, 1)
    y_t = df_c["azucar_producida_total"].astype(float).values
    n_train_t = int(round(len(df_c) * 0.80))  # 8 train, 2 test

    lr_t_eval = LinearRegression().fit(X_t[:n_train_t], y_t[:n_train_t])
    pred_t_train = lr_t_eval.predict(X_t[:n_train_t])
    pred_t_test = lr_t_eval.predict(X_t[n_train_t:])

    lr_t_full = LinearRegression().fit(X_t, y_t)
    pred_t_full = lr_t_full.predict(X_t)
    fut_x = np.array([[len(df_c) + 1], [len(df_c) + 2], [len(df_c) + 3]], dtype=float)
    fut_preds = lr_t_full.predict(fut_x)

    ts_metrics = {
        "n_zafras_completas": int(len(df_c)),
        "n_train": int(n_train_t),
        "n_test": int(len(df_c) - n_train_t),
        "train_zafras": f"{df_c['zafra'].iloc[0]} a {df_c['zafra'].iloc[n_train_t - 1]}",
        "test_zafras": f"{df_c['zafra'].iloc[n_train_t]} a {df_c['zafra'].iloc[-1]}",
        "ultimo_valor_real_zafra_completa": float(y_t[-1]),
        "ultima_zafra_completa": str(df_c["zafra"].iloc[-1]),
        "ultimo_valor_parcial_25_26": float(df_p["azucar_producida_total"].iloc[0]) if len(df_p) else 0.0,
        "semana_parcial_25_26": int(df_p["semana_cierre_max"].iloc[0]) if len(df_p) else 0,
        "ingenios_parcial_25_26": int(df_p["num_ingenios"].iloc[0]) if len(df_p) else 0,
        "max_zafra": str(df_c.loc[df_c["azucar_producida_total"].idxmax(), "zafra"]),
        "max_azucar": float(df_c["azucar_producida_total"].max()),
        "min_zafra": str(df_c.loc[df_c["azucar_producida_total"].idxmin(), "zafra"]),
        "min_azucar": float(df_c["azucar_producida_total"].min()),
        "beta0_eval_80": float(lr_t_eval.intercept_),
        "beta1_eval_80": float(lr_t_eval.coef_[0]),
        "r2_train": float(r2_score(y_t[:n_train_t], pred_t_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_t[:n_train_t], pred_t_train))),
        "mae_train": float(mean_absolute_error(y_t[:n_train_t], pred_t_train)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_t[n_train_t:], pred_t_test))),
        "mae_test": float(mean_absolute_error(y_t[n_train_t:], pred_t_test)),
        "beta0_full": float(lr_t_full.intercept_),
        "beta1_full": float(lr_t_full.coef_[0]),
        "r2_full": float(r2_score(y_t, pred_t_full)),
        "rmse_full": float(np.sqrt(mean_squared_error(y_t, pred_t_full))),
        "pronostico_2025_2026": float(fut_preds[0]),
        "pronostico_2026_2027": float(fut_preds[1]),
        "pronostico_2027_2028": float(fut_preds[2]),
    }

    # ------------------------------------------------------------------
    # 2. Regresión Lineal Operativa: azucar_producida_total ~ X (80/20 cronológico)
    # ------------------------------------------------------------------
    df_op = df_ing[(df_ing["cana_molida_neta"] > 0) & (df_ing["cana_molida_bruta"] > 0) & (df_ing["superficie_cosechada"] > 0) & (df_ing["azucar_producida_total"] > 0)].sort_values(["anio_inicio", "ingenio"]).reset_index(drop=True)
    n_op_tr = int(round(len(df_op) * 0.80))
    op_tr = df_op.iloc[:n_op_tr]
    op_te = df_op.iloc[n_op_tr:]

    op_models = {}
    for col in ["cana_molida_neta", "cana_molida_bruta", "superficie_cosechada"]:
        m = LinearRegression().fit(op_tr[[col]].values, op_tr["azucar_producida_total"].values)
        p_tr = m.predict(op_tr[[col]].values)
        p_te = m.predict(op_te[[col]].values)
        op_models[col] = {
            "n_train": int(len(op_tr)),
            "n_test": int(len(op_te)),
            "beta0": float(m.intercept_),
            "beta1": float(m.coef_[0]),
            "r2_train": float(r2_score(op_tr["azucar_producida_total"], p_tr)),
            "r2_test": float(r2_score(op_te["azucar_producida_total"], p_te)),
            "rmse_test": float(np.sqrt(mean_squared_error(op_te["azucar_producida_total"], p_te))),
            "mae_test": float(mean_absolute_error(op_te["azucar_producida_total"], p_te)),
        }

    # Modelo múltiple de referencia (caña neta + superficie)
    m_multi = LinearRegression().fit(op_tr[["cana_molida_neta", "superficie_cosechada"]].values, op_tr["azucar_producida_total"].values)
    p_te_multi = m_multi.predict(op_te[["cana_molida_neta", "superficie_cosechada"]].values)
    op_models["multiple_cana_superficie"] = {
        "n_train": int(len(op_tr)),
        "n_test": int(len(op_te)),
        "beta0": float(m_multi.intercept_),
        "coefs": [float(c) for c in m_multi.coef_],
        "r2_test": float(r2_score(op_te["azucar_producida_total"], p_te_multi)),
        "rmse_test": float(np.sqrt(mean_squared_error(op_te["azucar_producida_total"], p_te_multi))),
        "mae_test": float(mean_absolute_error(op_te["azucar_producida_total"], p_te_multi)),
    }

    metrics = {"serie_temporal": ts_metrics, "regresion_operativa": op_models}
    (PROC_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Gráfica 1: Serie Histórica Anual y Pronóstico (2015-2028)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafcff")
    labels_all = df_c["zafra"].tolist() + ["2025-2026*", "2026-2027*", "2027-2028*"]
    idx_h = np.arange(len(df_c))
    idx_f = np.arange(len(df_c), len(labels_all))
    y_h = y_t / 1e6
    y_fit = pred_t_full / 1e6
    y_f = fut_preds / 1e6
    band = ts_metrics["rmse_full"] / 1e6

    ax.axvspan(n_train_t - 0.5, len(df_c) - 1, color="#fff7ed", alpha=0.9, label="Prueba cronológica (20%)")
    ax.plot(idx_h, y_h, color="#1d6fd8", marker="o", linewidth=2.4, markersize=6.5, label="Datos históricos (Infocaña)")
    ax.plot(idx_h, y_fit, color="#64748b", linestyle="--", linewidth=1.4, label="Tendencia lineal OLS")
    ax.plot(np.concatenate([[idx_h[-1]], idx_f]), np.concatenate([[y_h[-1]], y_f]), color="#16a34a", marker="o", linewidth=2.5, markersize=6.8, label="Pronóstico lineal")
    ax.fill_between(idx_f, y_f - band, y_f + band, color="#22c55e", alpha=0.16, label="Banda ±1 RMSE")
    ax.axvline(idx_h[-1], color="#64748b", linestyle="--", linewidth=1.1)

    ax.annotate(
        f"Último dato real\n(2024-2025): {y_h[-1]:.2f} M ton",
        xy=(idx_h[-1], y_h[-1]),
        xytext=(idx_h[-1] - 2.5, y_h[-1] + 1.05),
        fontsize=8.5,
        fontweight="bold",
        color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f1fb", edgecolor="#93c5fd"),
        arrowprops=dict(arrowstyle="->", color="#1d6fd8"),
    )
    ax.annotate(
        f"Próx. zafra (25-26*):\n{y_f[0]:.2f} M ton",
        xy=(idx_f[0], y_f[0]),
        xytext=(idx_f[0] - 0.1, y_f[0] + 0.65),
        fontsize=8.5,
        fontweight="bold",
        color="#14532d",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#dcfce7", edgecolor="#86efac"),
        arrowprops=dict(arrowstyle="->", color="#16a34a"),
    )
    ax.set_xticks(np.arange(len(labels_all)))
    ax.set_xticklabels(labels_all, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Azúcar producida total (Millones de toneladas)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Ciclo / Zafra azucarera", fontsize=9.5, fontweight="bold")
    ax.set_title("Producción Total de Azúcar en México (2015-2025) y Pronóstico Lineal", fontsize=11, fontweight="bold", color="#1b365d")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(3.4, 7.0)
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", fontsize=8.0, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_serie_temporal_pronostico.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Gráfica 2: Regresión Operativa Y = B0 + B1 * Caña Molida Neta
    # ------------------------------------------------------------------
    cn_info = op_models["cana_molida_neta"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafcff")
    ax.scatter(op_tr["cana_molida_neta"] / 1e3, op_tr["azucar_producida_total"] / 1e3, color="#2563eb", alpha=0.55, s=32, label=f"Entrenamiento 80% (n={len(op_tr)})")
    ax.scatter(op_te["cana_molida_neta"] / 1e3, op_te["azucar_producida_total"] / 1e3, color="#f59e0b", alpha=0.82, s=38, edgecolors="#92400e", linewidths=0.5, label=f"Prueba 20% (n={len(op_te)})")

    x_line = np.linspace(0, df_op["cana_molida_neta"].max() * 1.02, 100)
    y_line = np.maximum(0, cn_info["beta0"] + cn_info["beta1"] * x_line)
    ax.plot(x_line / 1e3, y_line / 1e3, color="#15803d", linewidth=2.4, label=f"Recta OLS: Y = {cn_info['beta0']:,.0f} + {cn_info['beta1']:.4f}·X (R² test = {cn_info['r2_test']:.4f})")

    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Caña molida neta por ingenio (miles de toneladas)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Azúcar producida total (miles de toneladas)", fontsize=9.5, fontweight="bold")
    ax.set_title("Regresión Lineal Simple Operativa: Azúcar Total vs. Caña Molida Neta (2015–2026)", fontsize=11, fontweight="bold", color="#1b365d")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_regresion_operativa.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
