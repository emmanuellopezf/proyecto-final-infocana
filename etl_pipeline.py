"""Pipeline ETL para el proyecto Infocaña (CONADESUCA) 2015-2026.

Etapas:
  1. Extract (Extracción): Lee los archivos CSV históricos (2015-2026) en data/raw/
     (descargados de la API CKAN de datos.gob.mx + archivo local 2025-2026).
  2. Transform (Transformación):
     - Normalización de nombres de columnas entre esquemas 2015-2025 y 2025-2026.
     - Normalización de nombres de ingenios (acentos, espacios y capitalización).
     - Conversión estricta de tipos numéricos y manejo transparente de valores nulos.
     - Generación de variables derivadas (rendimientos, tasas de extracción,
       rendimiento promedio histórico por ingenio).
     - Agregación por (zafra, ingenio) tomando la semana de cierre (datos acumulados),
       agregación anual nacional por zafra y serie mensual incremental.
  3. Load (Carga):
     - Exporta datasets limpios en CSV (data/processed/ y app/data/ para ShinyLive).
     - Persiste las tablas consolidadas en SQLite (data/processed/infocana.db).
     - Genera auditoría JSON de calidad de datos (data/processed/etl_summary.json).
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
APP_DATA_DIR = BASE_DIR / "app" / "data"

COLUMN_RENAME_MAP = {
    "semana": "no_s_zafra",
    "azucar_total": "azucar_producida_total",
    "azucar_refinada": "azucar_producida_refinada",
    "azucar_estandar": "azucar_producida_estandar",
    "azucar_blanca_especial": "azucar_producida_blanca_especial",
    "azucar_mascabado": "azucar_producida_mascabado",
    "azucar_pol_menor_99_2": "toneladas_azucar_con_pol_menor_99_2",
    "karbe_cana_bruta": "karbe_ton_cana_bruta_teorico",
    "karbe_cana_neta": "karbe_ton_cania_neta_teorico",
    "eficiencia_fabrica": "eficiencia_en_fabrica",
}

CANONICAL_COLS = [
    "ingenio",
    "zafra",
    "no_s_zafra",
    "cana_molida_bruta",
    "cana_molida_neta",
    "superficie_cosechada",
    "azucar_producida_total",
    "azucar_producida_refinada",
    "azucar_producida_estandar",
    "azucar_producida_blanca_especial",
    "azucar_producida_mascabado",
    "toneladas_azucar_con_pol_menor_99_2",
    "karbe_ton_cana_bruta_teorico",
    "karbe_ton_cania_neta_teorico",
    "eficiencia_en_fabrica",
    "rendimiento_campo",
    "rendimiento_fabrica",
    "rendimiento_agroindustrial",
]

NUMERIC_COLS = [c for c in CANONICAL_COLS if c not in ("ingenio", "zafra")]

# Mapeo de meses aproximados por semana de zafra (la zafra inicia en octubre: semana 1 ~ oct)
# Semana 1-4: Oct, 5-8: Nov, 9-13: Dic, 14-17: Ene, 18-21: Feb, 22-26: Mar,
# 27-30: Abr, 31-35: May, 36-39: Jun, 40-44: Jul, 45+: Ago
def semana_a_anio_mes(anio_inicio: int, semana: int) -> str:
    # Mes offset desde octubre (mes 10 de anio_inicio)
    mes_offset = min(int((semana - 1) // 4.33), 11)
    total_mes = 10 + mes_offset
    anio = anio_inicio + (total_mes - 1) // 12
    mes = ((total_mes - 1) % 12) + 1
    return f"{anio:04d}-{mes:02d}"


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_key(name: str) -> str:
    clean = _strip_accents(str(name)).lower().strip()
    clean = re.sub(r"\s*-\s*siembra$", "", clean)
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def build_ingenio_canonical_map(raw_files: list[Path]) -> dict[str, str]:
    """Construye catálogo canónico de ingenios usando los nombres oficiales con acentos de 2025-2026."""
    canon: dict[str, str] = {}
    file_25_26 = RAW_DIR / "infocana_2025-2026.csv"
    if file_25_26.exists():
        df25 = pd.read_csv(file_25_26, encoding="utf-8-sig")
        for raw_name in df25["ingenio"].dropna().unique():
            clean_display = re.sub(r"\s*-\s*Siembra$", "", str(raw_name).strip(), flags=re.IGNORECASE)
            key = normalize_key(clean_display)
            if key and key not in canon:
                canon[key] = clean_display

    # Correcciones manuales conocidas de ingenios históricos que no aparecen en semanas 7-14 de 2025-2026
    known_accents = {
        "aaron saenz garza": "Aarón Sáenz Garza",
        "azsuremex": "Azsuremex",
        "azsuremex tenosique": "Azsuremex",
        "benito juarez": "Presidente Benito Juárez",
        "calipam": "Calipam",
        "central progreso": "Central Progreso",
        "ciasa cuatotolapam": "Cuatotolapam",
        "cuatotolapam": "Cuatotolapam",
        "el carmen": "El Carmen",
        "el dorado": "Eldorado",
        "eldorado": "Eldorado",
        "el refugio": "El Refugio",
        "el potrero": "Central El Potrero",
        "emiliano zapata": "Emiliano Zapata",
        "jose ma martinez tala": "Tala",
        "jose ma morelos": "José María Morelos",
        "la providencia": "Central La Providencia",
        "la providencia central": "Central La Providencia",
        "lopez mateos": "Adolfo López Mateos",
        "mahuitlan": "Mahuixtlán",
        "mahuixtlan": "Mahuixtlán",
        "melchor ocampo": "Melchor Ocampo",
        "motzorongo": "Central Motzorongo",
        "nuevo san francisco": "Nuevo San Francisco",
        "pablo machado la margarita": "La Margarita",
        "panuco": "Pánuco",
        "presidente benito juarez": "Presidente Benito Juárez",
        "presidente benito juarez santos": "Presidente Benito Juárez",
        "pujiltic": "Cía. La Fe (Pujiltic)",
        "pujiltic cia la fe": "Cía. La Fe (Pujiltic)",
        "cia la fe pujiltic": "Cía. La Fe (Pujiltic)",
        "queseria": "Quesería",
        "san cristobal": "San Cristóbal",
        "san cristobal zcmx": "San Cristóbal",
        "san francisco el naranjal": "San Francisco El Naranjal",
        "san jose de abajo": "San José de Abajo",
        "san miguelito": "Central San Miguelito",
        "san nicola s": "San Nicolás",
        "san nicolas": "San Nicolás",
        "san rafael de pucte": "San Rafael de Pucté",
        "santa rosalia": "Santa Rosalía",
        "tamazula": "Tamazula",
        "tamazula saenz": "Tamazula",
        "zapoapita": "Zapoapita",
    }
    for k, v in known_accents.items():
        canon.setdefault(k, v)
    return canon


def extract_and_normalize() -> tuple[pd.DataFrame, dict]:
    raw_files = sorted(RAW_DIR.glob("infocana_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {RAW_DIR}. Ejecuta scripts/download_raw.py primero.")

    canon_map = build_ingenio_canonical_map(raw_files)
    frames: list[pd.DataFrame] = []
    file_stats: list[dict] = []

    for path in raw_files:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        raw_rows = len(df)
        raw_cols = len(df.columns)

        # Normalizar nombres de columnas
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns=COLUMN_RENAME_MAP)

        # Inferir zafra desde el nombre del archivo si hubiera faltantes
        zafra_from_name = path.stem.replace("infocana_", "")
        if "zafra" not in df.columns:
            df["zafra"] = zafra_from_name
        else:
            df["zafra"] = df["zafra"].fillna(zafra_from_name).astype(str).str.strip()

        # Asegurar todas las columnas canónicas
        for col in CANONICAL_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[CANONICAL_COLS].copy()
        frames.append(df)
        file_stats.append(
            {
                "archivo": path.name,
                "zafra": zafra_from_name,
                "registros_crudos": int(raw_rows),
                "columnas_crudas": int(raw_cols),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    raw_total_rows = len(combined)
    nulls_before = {col: int(combined[col].isna().sum()) for col in CANONICAL_COLS}

    # Limpieza de columnas categóricas
    combined["ingenio"] = combined["ingenio"].astype(str).str.strip()
    combined = combined[combined["ingenio"].ne("") & combined["ingenio"].ne("nan")].copy()
    combined["ingenio_key"] = combined["ingenio"].map(normalize_key)
    combined["ingenio"] = combined["ingenio_key"].map(
        lambda k: canon_map.get(k, " ".join(w.capitalize() for w in k.split()))
    )

    # Normalizar formato de zafra (YYYY-YYYY)
    combined["zafra"] = combined["zafra"].astype(str).str.strip().str.replace("/", "-", regex=False)
    valid_zafra_mask = combined["zafra"].str.match(r"^\d{4}-\d{4}$")
    combined = combined[valid_zafra_mask].copy()

    # Conversión de tipos numéricos
    for col in NUMERIC_COLS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Eliminar registros sin variables núcleo (semana, caña bruta o azúcar total)
    core_required = ["no_s_zafra", "azucar_producida_total", "cana_molida_bruta", "superficie_cosechada"]
    rows_before_core_drop = len(combined)
    combined = combined.dropna(subset=core_required).copy()
    dropped_core_nulls = rows_before_core_drop - len(combined)

    # Eliminar valores negativos anómalos en variables de producción
    combined = combined[
        (combined["no_s_zafra"] >= 1)
        & (combined["azucar_producida_total"] >= 0)
        & (combined["cana_molida_bruta"] >= 0)
        & (combined["superficie_cosechada"] >= 0)
    ].copy()

    # Imputación contextual de subtipos opcionales de azúcar (cuando no se produjo ese subtipo = 0.0)
    subtype_cols = [
        "azucar_producida_refinada",
        "azucar_producida_estandar",
        "azucar_producida_blanca_especial",
        "azucar_producida_mascabado",
        "toneladas_azucar_con_pol_menor_99_2",
        "karbe_ton_cana_bruta_teorico",
        "karbe_ton_cania_neta_teorico",
        "eficiencia_en_fabrica",
    ]
    for col in subtype_cols:
        combined[col] = combined[col].fillna(0.0)

    # Si cana_molida_neta es nula o 0 cuando hay caña bruta, imputar usando proporción histórica mediana (neta/bruta ~ 0.955)
    valid_ratio = combined.loc[
        (combined["cana_molida_neta"] > 0) & (combined["cana_molida_bruta"] > 0),
        "cana_molida_neta",
    ] / combined.loc[
        (combined["cana_molida_neta"] > 0) & (combined["cana_molida_bruta"] > 0),
        "cana_molida_bruta",
    ]
    median_neta_ratio = float(valid_ratio.median()) if len(valid_ratio) else 0.955
    mask_missing_neta = combined["cana_molida_neta"].isna() | (
        (combined["cana_molida_neta"] == 0) & (combined["cana_molida_bruta"] > 0)
    )
    imputed_neta_count = int(mask_missing_neta.sum())
    combined.loc[mask_missing_neta, "cana_molida_neta"] = (
        combined.loc[mask_missing_neta, "cana_molida_bruta"] * median_neta_ratio
    )

    # Entero para semana
    combined["no_s_zafra"] = combined["no_s_zafra"].round().astype(int)

    # Deduplicar (zafra, ingenio, no_s_zafra)
    before_dedup = len(combined)
    combined = combined.sort_values(["zafra", "ingenio", "no_s_zafra", "azucar_producida_total"]).drop_duplicates(
        subset=["zafra", "ingenio", "no_s_zafra"], keep="last"
    )
    duplicates_removed = before_dedup - len(combined)

    # Variables derivadas temporales
    combined["anio_inicio"] = combined["zafra"].str.slice(0, 4).astype(int)
    combined["anio_fin"] = combined["zafra"].str.slice(5, 9).astype(int)
    zafras_sorted = sorted(combined["zafra"].unique())
    zafra_to_idx = {z: i + 1 for i, z in enumerate(zafras_sorted)}
    combined["zafra_idx"] = combined["zafra"].map(zafra_to_idx)
    combined["periodo_mes"] = [
        semana_a_anio_mes(ai, sem) for ai, sem in zip(combined["anio_inicio"], combined["no_s_zafra"])
    ]

    # Recalcular/completar variables derivadas agronómicas e industriales
    combined["rendimiento_campo"] = np.where(
        combined["superficie_cosechada"] > 0,
        combined["cana_molida_bruta"] / combined["superficie_cosechada"],
        0.0,
    )
    combined["rendimiento_fabrica"] = np.where(
        combined["cana_molida_bruta"] > 0,
        (combined["azucar_producida_total"] / combined["cana_molida_bruta"]) * 100.0,
        0.0,
    )
    combined["rendimiento_agroindustrial"] = np.where(
        combined["superficie_cosechada"] > 0,
        combined["azucar_producida_total"] / combined["superficie_cosechada"],
        0.0,
    )
    combined["tasa_extraccion_neta_pct"] = np.where(
        combined["cana_molida_neta"] > 0,
        (combined["azucar_producida_total"] / combined["cana_molida_neta"]) * 100.0,
        0.0,
    )

    # Marcar estado de la zafra (las zafras completas superan la semana 30)
    max_week_by_zafra = combined.groupby("zafra")["no_s_zafra"].max().to_dict()
    combined["estado_zafra"] = combined["zafra"].map(
        lambda z: "Completa" if max_week_by_zafra[z] >= 30 else f"En curso (sem. {max_week_by_zafra[z]})"
    )

    combined = combined.drop(columns=["ingenio_key"]).sort_values(["zafra", "ingenio", "no_s_zafra"]).reset_index(drop=True)

    audit = {
        "archivos_procesados": len(raw_files),
        "detalle_archivos": file_stats,
        "registros_crudos_totales": int(raw_total_rows),
        "registros_limpios_semanales": int(len(combined)),
        "registros_eliminados_nulos_nucleo": int(dropped_core_nulls),
        "duplicados_eliminados": int(duplicates_removed),
        "imputaciones_cana_neta": int(imputed_neta_count),
        "ratio_mediana_neta_bruta": round(median_neta_ratio, 4),
        "nulos_antes_limpieza": nulls_before,
        "nulos_despues_limpieza": {col: int(combined[col].isna().sum()) for col in CANONICAL_COLS},
        "zafras_totales": len(zafras_sorted),
        "rango_zafras": f"{zafras_sorted[0]} a {zafras_sorted[-1]}",
        "ingenios_unicos": int(combined["ingenio"].nunique()),
    }
    return combined, audit


def build_aggregations(df_weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye las tablas agregadas evitando doble conteo de datos acumulados."""
    # 1. Cierre por (zafra, ingenio): fila con la semana máxima registrada por ingenio en cada zafra
    idx_cierre = df_weekly.groupby(["zafra", "ingenio"])["no_s_zafra"].idxmax()
    df_ingenio_zafra = df_weekly.loc[idx_cierre].copy().sort_values(["zafra", "ingenio"]).reset_index(drop=True)

    # Variable derivada: rendimiento agroindustrial promedio histórico por ingenio (solo zafras completas)
    completas_ing = df_ingenio_zafra[df_ingenio_zafra["estado_zafra"] == "Completa"]
    rend_prom_ingenio = (
        completas_ing.groupby("ingenio")["rendimiento_agroindustrial"]
        .mean()
        .rename("rendimiento_promedio_ingenio")
    )
    df_ingenio_zafra = df_ingenio_zafra.merge(rend_prom_ingenio, on="ingenio", how="left")
    df_ingenio_zafra["rendimiento_promedio_ingenio"] = df_ingenio_zafra["rendimiento_promedio_ingenio"].fillna(
        df_ingenio_zafra["rendimiento_agroindustrial"]
    )

    # 2. Resumen nacional anual por zafra
    df_zafra = (
        df_ingenio_zafra.groupby(["zafra", "anio_inicio", "anio_fin", "zafra_idx", "estado_zafra"], as_index=False)
        .agg(
            num_ingenios=("ingenio", "nunique"),
            semana_cierre_max=("no_s_zafra", "max"),
            cana_molida_bruta=("cana_molida_bruta", "sum"),
            cana_molida_neta=("cana_molida_neta", "sum"),
            superficie_cosechada=("superficie_cosechada", "sum"),
            azucar_producida_total=("azucar_producida_total", "sum"),
            azucar_producida_refinada=("azucar_producida_refinada", "sum"),
            azucar_producida_estandar=("azucar_producida_estandar", "sum"),
            azucar_producida_blanca_especial=("azucar_producida_blanca_especial", "sum"),
            azucar_producida_mascabado=("azucar_producida_mascabado", "sum"),
            eficiencia_en_fabrica_prom=("eficiencia_en_fabrica", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        )
        .sort_values("anio_inicio")
        .reset_index(drop=True)
    )
    df_zafra["rendimiento_campo"] = df_zafra["cana_molida_bruta"] / df_zafra["superficie_cosechada"]
    df_zafra["rendimiento_fabrica"] = (df_zafra["azucar_producida_total"] / df_zafra["cana_molida_bruta"]) * 100.0
    df_zafra["rendimiento_agroindustrial"] = df_zafra["azucar_producida_total"] / df_zafra["superficie_cosechada"]
    df_zafra["azucar_millones_ton"] = df_zafra["azucar_producida_total"] / 1_000_000.0
    df_zafra["cana_neta_millones_ton"] = df_zafra["cana_molida_neta"] / 1_000_000.0

    # 3. Serie semanal nacional (avance acumulado por semana dentro de cada zafra)
    # Para cada (zafra, semana), se toma el último valor acumulado conocido de cada ingenio hasta esa semana (forward-fill)
    weekly_nat_rows = []
    for zafra, grp in df_weekly.groupby("zafra"):
        semanas = sorted(grp["no_s_zafra"].unique())
        anio_inicio = int(grp["anio_inicio"].iloc[0])
        estado = str(grp["estado_zafra"].iloc[0])
        piv_az = grp.pivot_table(index="no_s_zafra", columns="ingenio", values="azucar_producida_total", aggfunc="max").sort_index().ffill().fillna(0.0)
        piv_cn = grp.pivot_table(index="no_s_zafra", columns="ingenio", values="cana_molida_neta", aggfunc="max").sort_index().ffill().fillna(0.0)
        piv_sup = grp.pivot_table(index="no_s_zafra", columns="ingenio", values="superficie_cosechada", aggfunc="max").sort_index().ffill().fillna(0.0)
        for sem in semanas:
            weekly_nat_rows.append(
                {
                    "zafra": zafra,
                    "anio_inicio": anio_inicio,
                    "estado_zafra": estado,
                    "no_s_zafra": int(sem),
                    "periodo_mes": semana_a_anio_mes(anio_inicio, int(sem)),
                    "ingenios_activos": int((piv_az.loc[sem] > 0).sum()),
                    "azucar_acumulada_ton": float(piv_az.loc[sem].sum()),
                    "cana_neta_acumulada_ton": float(piv_cn.loc[sem].sum()),
                    "superficie_acumulada_ha": float(piv_sup.loc[sem].sum()),
                }
            )
    df_semanal_nacional = pd.DataFrame(weekly_nat_rows).sort_values(["anio_inicio", "no_s_zafra"]).reset_index(drop=True)
    # Producción incremental semanal
    df_semanal_nacional["azucar_semanal_ton"] = (
        df_semanal_nacional.groupby("zafra")["azucar_acumulada_ton"].diff().fillna(df_semanal_nacional["azucar_acumulada_ton"]).clip(lower=0.0)
    )

    return df_ingenio_zafra, df_zafra, df_semanal_nacional


def load_outputs(
    df_weekly: pd.DataFrame,
    df_ingenio_zafra: pd.DataFrame,
    df_zafra: pd.DataFrame,
    df_semanal_nacional: pd.DataFrame,
    audit: dict,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Redondeo limpio para archivos CSV
    num_round_cols = [c for c in df_ingenio_zafra.select_dtypes(include="number").columns if c not in ("no_s_zafra", "anio_inicio", "anio_fin", "zafra_idx")]
    df_ing_out = df_ingenio_zafra.copy()
    df_ing_out[num_round_cols] = df_ing_out[num_round_cols].round(3)

    df_zafra_out = df_zafra.copy()
    for c in df_zafra_out.select_dtypes(include="float").columns:
        df_zafra_out[c] = df_zafra_out[c].round(3)

    df_sem_out = df_semanal_nacional.copy()
    for c in df_sem_out.select_dtypes(include="float").columns:
        df_sem_out[c] = df_sem_out[c].round(2)

    # Guardar en data/processed/
    df_weekly.to_csv(PROCESSED_DIR / "infocana_semanal_limpio.csv", index=False, encoding="utf-8")
    df_ing_out.to_csv(PROCESSED_DIR / "infocana_ingenio_zafra.csv", index=False, encoding="utf-8")
    df_zafra_out.to_csv(PROCESSED_DIR / "infocana_resumen_zafra.csv", index=False, encoding="utf-8")
    df_sem_out.to_csv(PROCESSED_DIR / "infocana_semanal_nacional.csv", index=False, encoding="utf-8")

    # Guardar datasets compactos dentro de app/data/ para empaquetado estático ShinyLive (GitHub Pages)
    df_ing_out.to_csv(APP_DATA_DIR / "infocana_ingenio_zafra.csv", index=False, encoding="utf-8")
    df_zafra_out.to_csv(APP_DATA_DIR / "infocana_resumen_zafra.csv", index=False, encoding="utf-8")
    df_sem_out.to_csv(APP_DATA_DIR / "infocana_semanal_nacional.csv", index=False, encoding="utf-8")

    # Guardar en base de datos SQLite (opcional recomendado en metodología)
    db_path = PROCESSED_DIR / "infocana.db"
    with sqlite3.connect(db_path) as conn:
        df_weekly.to_sql("infocana_semanal_limpio", conn, if_exists="replace", index=False)
        df_ing_out.to_sql("infocana_ingenio_zafra", conn, if_exists="replace", index=False)
        df_zafra_out.to_sql("infocana_resumen_zafra", conn, if_exists="replace", index=False)
        df_sem_out.to_sql("infocana_semanal_nacional", conn, if_exists="replace", index=False)

    # Guardar auditoría ETL
    audit["registros_cierre_ingenio_zafra"] = int(len(df_ing_out))
    audit["registros_resumen_zafra"] = int(len(df_zafra_out))
    audit["registros_semanal_nacional"] = int(len(df_sem_out))
    (PROCESSED_DIR / "etl_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (APP_DATA_DIR / "etl_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    print("=== Ejecutando Pipeline ETL Infocaña (2015-2026) ===")
    df_weekly, audit = extract_and_normalize()
    df_ingenio_zafra, df_zafra, df_semanal_nacional = build_aggregations(df_weekly)
    load_outputs(df_weekly, df_ingenio_zafra, df_zafra, df_semanal_nacional, audit)

    print(f"Archivos procesados       : {audit['archivos_procesados']}")
    print(f"Registros crudos totales  : {audit['registros_crudos_totales']:,}")
    print(f"Registros semanales limp. : {audit['registros_limpios_semanales']:,}")
    print(f"Registros cierre ing/zafra: {audit['registros_cierre_ingenio_zafra']:,}")
    print(f"Zafras consolidadas       : {audit['rango_zafras']} ({audit['zafras_totales']} zafras)")
    print(f"Ingenios únicos           : {audit['ingenios_unicos']}")
    print("\n--- Resumen Nacional por Zafra ---")
    cols_show = ["zafra", "estado_zafra", "num_ingenios", "semana_cierre_max", "azucar_producida_total", "cana_molida_neta", "rendimiento_agroindustrial"]
    print(df_zafra[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
