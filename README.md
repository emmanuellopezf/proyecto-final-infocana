# Análisis Predictivo de la Producción de Azúcar en México (Infocaña 2015–2026)

**Dashboard en vivo (GitHub Pages / ShinyLive):** [https://emmanuellopezf.github.io/proyecto-final-infocana/](https://emmanuellopezf.github.io/proyecto-final-infocana/)  
**Repositorio:** [https://github.com/emmanuellopezf/proyecto-final-infocana](https://github.com/emmanuellopezf/proyecto-final-infocana)  
**Unidad de Competencia:** Big Data  
**Programa:** Ingeniería en Desarrollo y Tecnologías de Software (LIDTS) — Benemérita UNACH  
**Docente:** Dr. Christian Mauricio Castillo Estrada  

---

## Descripción del Proyecto

Este proyecto integra un **pipeline ETL completo en Python** y un **Dashboard interactivo en Shiny for Python** (desplegable de forma estática en **GitHub Pages** mediante **ShinyLive**) para analizar y pronosticar la producción total de azúcar en México con base en los datos abiertos de **Infocaña (CONADESUCA)** correspondientes a las zafras **2015–2026**.

Fuente oficial de datos: [datos.gob.mx — Avance de la producción de caña y azúcar (Infocaña)](https://www.datos.gob.mx/dataset/avance_produccion_cana_azucar_infocana).

---

## Estructura del Repositorio

```text
proyecto_final/
├── Infocana_25_26_resumen.csv        # Dataset original proporcionado (zafra 2025-2026)
├── ProyectoFinal_AgoDic26.pdf        # Especificación oficial del proyecto
├── etl_pipeline.py                   # Pipeline ETL (Extract, Transform, Load + SQLite)
├── requirements.txt                  # Dependencias Python del proyecto
├── scripts/
│   ├── download_raw.py               # Extracción automatizada desde API CKAN datos.gob.mx (2015-2026)
│   ├── evaluate_and_plot.py          # Evaluación 80%/20% de modelos sklearn y gráficas
│   └── generate_report_pdf.py        # Generador del Informe Final (.pdf)
├── data/
│   ├── raw/                          # 11 archivos CSV crudos (2015-2026)
│   └── processed/                    # Tablas limpias (.csv), SQLite (infocana.db) y auditorías (.json)
├── app/
│   ├── app.py                        # Aplicación Shiny for Python
│   └── data/                         # Datasets procesados empaquetados para ShinyLive
├── docs/                             # Sitio estático generado con `shinylive export app docs` (GitHub Pages)
└── reports/
    ├── figures/                      # Gráficas de evaluación y capturas reales del Dashboard
    └── Informe_Proyecto_Final_Infocana.pdf  # Informe Final entregable en PDF
```

---

## Ejecución Local

1. Crear y activar el entorno virtual e instalar dependencias:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip.exe install -r requirements.txt
   ```

2. Ejecutar extracción y pipeline ETL:
   ```powershell
   .\.venv\Scripts\python.exe scripts/download_raw.py
   .\.venv\Scripts\python.exe etl_pipeline.py
   ```

3. Ejecutar el Dashboard interactivo localmente:
   ```powershell
   .\.venv\Scripts\shiny.exe run app/app.py --reload
   ```

4. Re-exportar la versión estática para GitHub Pages (ShinyLive):
   ```powershell
   .\.venv\Scripts\shinylive.exe export app docs
   ```

---

## Despliegue en GitHub Pages (Paso a Paso — Automático con GitHub Actions)

El proyecto incluye el flujo automatizado `.github/workflows/deploy-pages.yml` que compila el Dashboard con **ShinyLive** directamente en los servidores de GitHub y lo publica en **GitHub Pages** sin necesidad de subir archivos pesados de WebAssembly.

1. Crea un repositorio **público** nuevo en GitHub (por ejemplo, `proyecto-final-infocana`).
2. Sube los archivos del proyecto (`app/`, `.github/`, `etl_pipeline.py`, `scripts/`, `README.md`, `requirements.txt` — **sin** la carpeta `.venv/`).
3. En tu repositorio de GitHub, ve a **Settings → Pages**.
4. En **Build and deployment → Source**, selecciona **GitHub Actions**.
5. Ve a la pestaña **Actions** y verifica que el workflow *Desplegar Dashboard ShinyLive en GitHub Pages* finalice en verde (~1 minuto).
6. Tu Dashboard quedará publicado y 100% interactivo en:
   `https://<tu-usuario>.github.io/<nombre-del-repositorio>/`

> **Opción alternativa local:** también puedes ejecutar `.\.venv\Scripts\shinylive.exe export app docs` localmente y configurar GitHub Pages desde la rama `main` y carpeta `/docs`.
