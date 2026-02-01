import subprocess
import atexit
import os
import signal

chromium_process = None

def start_chromium():
    """Launch Chromium in kiosk mode pointing to our Flask app"""
    global chromium_process
    subprocess.run(["pkill", "-f", "chromium"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chromium_process = subprocess.Popen([
        "chromium",
        "--kiosk",
        "--noerrdialogs",
        "--disable-infobars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-background-networking",
        "http://localhost:5000"
    ], env={**os.environ, "DISPLAY": ":0"})

def test_start_chromium():
    """Launch Chromium in kiosk mode pointing to our Flask app"""
    global chromium_process
    subprocess.run(["pkill", "-f", "chromium"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chromium_process = subprocess.Popen([
        "chromium",
        "--kiosk",
        "--noerrdialogs",
        "--disable-infobars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-renderer-backgrounding",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-features=TranslateUI",
        "--disable-features=BackForwardCache",
        "--disable-dev-shm-usage",
        "--disable-gpu-vsync",
        "--enable-low-end-device-mode",
        "--use-gl=egl",
        "--enable-accelerated-2d-canvas",
        "--disable-features=UseOzonePlatform",
        "--autoplay-policy=no-user-gesture-required",
        "http://127.0.0.1:5000"
    ], env={**os.environ, "DISPLAY": ":0"})

def stop_chromium():
    """Kill Chromium when Python exits"""
    global chromium_process
    if chromium_process:
        print("[INFO] Stopping Chromium...")
        try:
            chromium_process.terminate()  # graceful shutdown
            chromium_process.wait(timeout=2)
        except Exception:
            chromium_process.kill()  # force kill if needed

atexit.register(stop_chromium)
