"""Inicia temporalmente el Dashboard Shiny, verifica HTTP 200 y captura
pantallazos reales en alta resolución usando Chrome Headless + CDP.
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SHINY_EXE = BASE_DIR / ".venv" / "Scripts" / "shiny.exe"
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
if not CHROME_EXE.exists():
    CHROME_EXE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def wait_http(url: str, timeout: float = 25.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"El servidor no respondió en {url}")


async def capture_tabs(cdp_port: int, app_url: str) -> None:
    targets_url = f"http://127.0.0.1:{cdp_port}/json"
    wait_http(targets_url, timeout=15.0)
    with urllib.request.urlopen(targets_url, timeout=5) as resp:
        targets = json.loads(resp.read().decode("utf-8"))
    page_target = next(t for t in targets if t.get("type") == "page")
    ws_url = page_target["webSocketDebuggerUrl"]

    msg_id = 0

    async with websockets.connect(ws_url, max_size=25 * 1024 * 1024) as ws:
        async def send_cdp(method: str, params: dict | None = None) -> dict:
            nonlocal msg_id
            msg_id += 1
            cid = msg_id
            await ws.send(json.dumps({"id": cid, "method": method, "params": params or {}}))
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("id") == cid:
                    return data

        await send_cdp("Page.enable")
        await send_cdp("Emulation.setDeviceMetricsOverride", {
            "width": 1440,
            "height": 1020,
            "deviceScaleFactor": 1.5,
            "mobile": False,
        })
        await send_cdp("Page.navigate", {"url": app_url})
        await asyncio.sleep(5.5)

        # Captura Tab 1 (Serie de Tiempo y Pronóstico)
        shot1 = await send_cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        (FIG_DIR / "dashboard_tab1_serie_tiempo.png").write_bytes(base64.b64decode(shot1["result"]["data"]))
        print("[OK] Captura Tab 1 guardada.")

        # Clic en Tab 2 (Regresión Operativa)
        await send_cdp("Runtime.evaluate", {
            "expression": "document.querySelectorAll('.nav-tabs .nav-link')[1].click();"
        })
        await asyncio.sleep(3.5)
        shot2 = await send_cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        (FIG_DIR / "dashboard_tab2_regresion_operativa.png").write_bytes(base64.b64decode(shot2["result"]["data"]))
        print("[OK] Captura Tab 2 guardada.")

        # Clic en Tab 3 (Explorador ETL e Ingenios)
        await send_cdp("Runtime.evaluate", {
            "expression": "document.querySelectorAll('.nav-tabs .nav-link')[2].click();"
        })
        await asyncio.sleep(3.0)
        shot3 = await send_cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        (FIG_DIR / "dashboard_tab3_explorador_etl.png").write_bytes(base64.b64decode(shot3["result"]["data"]))
        print("[OK] Captura Tab 3 guardada.")

        # Verificar que no haya errores visibles de Shiny en DOM
        err_check = await send_cdp("Runtime.evaluate", {
            "expression": "document.querySelectorAll('.shiny-output-error').length",
            "returnByValue": True,
        })
        err_count = err_check.get("result", {}).get("result", {}).get("value", -1)
        print(f"[CHECK] Errores visibles de Shiny en DOM: {err_count}")


def main() -> None:
    app_port = 8765
    cdp_port = 9229
    shiny_proc = subprocess.Popen(
        [str(SHINY_EXE), "run", str(BASE_DIR / "app" / "app.py"), "--host", "127.0.0.1", "--port", str(app_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    chrome_proc = None
    try:
        wait_http(f"http://127.0.0.1:{app_port}/", timeout=25.0)
        print(f"[OK] Shiny server activo en http://127.0.0.1:{app_port}/")
        user_data_dir = BASE_DIR / "reports" / ".chrome_tmp"
        user_data_dir.mkdir(parents=True, exist_ok=True)
        chrome_proc = subprocess.Popen(
            [
                str(CHROME_EXE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--remote-debugging-port={cdp_port}",
                f"--user-data-dir={user_data_dir}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        asyncio.run(capture_tabs(cdp_port, f"http://127.0.0.1:{app_port}/"))
    finally:
        if chrome_proc:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
        shiny_proc.terminate()
        try:
            shiny_proc.wait(timeout=5)
        except Exception:
            shiny_proc.kill()


if __name__ == "__main__":
    main()
