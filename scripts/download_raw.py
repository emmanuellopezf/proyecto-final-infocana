"""Extracción (Extract) de los datasets históricos de Infocaña (CONADESUCA) 2015-2026
desde la API oficial CKAN DataStore de datos.gob.mx y el archivo local 2025-2026.
"""
from pathlib import Path
import csv
import json
import shutil
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API = "https://www.datos.gob.mx/api/3/action/datastore_search"

# Resource IDs oficiales publicados en:
# https://www.datos.gob.mx/dataset/avance_produccion_cana_azucar_infocana
RESOURCES = {
    "2015-2016": "9808cfad-c774-4fbb-9cd6-469781ec5479",
    "2016-2017": "d1fe9381-4acf-4e06-9350-1fa406a7094e",
    "2017-2018": "7aec32a8-ac2e-4db3-b8bb-368ab6bb553e",
    "2018-2019": "cca6d3b3-a375-4b9c-9dbe-0a911cd3c805",
    "2019-2020": "eb7afc9f-096c-4f22-803f-28ea715c6b52",
    "2020-2021": "ffae7ef9-6432-4dc7-940f-0c58bfba0f6d",
    "2021-2022": "cc37b46c-2e5b-423b-9df2-36153aded68c",
    "2022-2023": "a20167cc-6f36-4eaa-b59f-874a05c3fc44",
    "2023-2024": "cb1e8afb-c31e-46d0-ac72-8d7ea1a233b2",
    "2024-2025": "4bddfc19-83fd-429b-aafe-d610a5ca6495",
    "2025-2026": "e87650e2-c1d8-41b1-b5e2-673cbad7b0cd",
}


def fetch_resource(resource_id: str) -> tuple[list[str], list[dict]]:
    records: list[dict] = []
    fields: list[str] = []
    offset = 0
    limit = 2000
    while True:
        url = f"{CKAN_API}?resource_id={resource_id}&limit={limit}&offset={offset}"
        req = urllib.request.Request(url, headers={"User-Agent": "InfocanaETL/1.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"Error CKAN API para {resource_id}: {payload}")
        result = payload["result"]
        if not fields:
            fields = [f["id"] for f in result["fields"] if f["id"] != "_id"]
        batch = result["records"]
        records.extend(batch)
        total = result.get("total", len(records))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return fields, records


def main() -> None:
    local_25_26 = BASE_DIR / "Infocana_25_26_resumen.csv"
    for zafra, rid in RESOURCES.items():
        dest = RAW_DIR / f"infocana_{zafra}.csv"
        if zafra == "2025-2026" and local_25_26.exists():
            shutil.copy2(local_25_26, dest)
            with dest.open(encoding="utf-8-sig") as fh:
                n_rows = sum(1 for _ in fh) - 1
            print(f"[LOCAL] {zafra} -> {dest.name} ({n_rows} registros)")
            continue
        fields, records = fetch_resource(rid)
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"[CKAN]  {zafra} -> {dest.name} ({len(records)} registros, {len(fields)} cols)")


if __name__ == "__main__":
    main()
