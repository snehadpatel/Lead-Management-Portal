#!/usr/bin/env python3
"""
Lume AI — Unified Full-Stack Runner
Launches both the FastAPI backend (port 8000) and the Frontend web server (port 3000) concurrently.
Handles graceful shutdown (Ctrl+C) to terminate both servers cleanly.

Usage:
    python3 run.py
    # or
    ./run.sh
    # or
    make run
"""

import os
import sys
import time
import signal
import socket
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_process_on_port(port: int):
    """Attempt to terminate any process listening on a port."""
    try:
        cmd = f"lsof -ti :{port} | xargs kill -9 2>/dev/null || true"
        subprocess.run(cmd, shell=True, check=False)
        time.sleep(0.5)
    except Exception:
        pass


def print_banner(backend_port: int, frontend_port: int):
    """Display startup banner with URLs."""
    banner = f"""
========================================================================
  💎  LUME AI — FINTECH PLATFORM RUNNER
========================================================================
  🌐 Frontend Web App : http://localhost:{frontend_port}
     ├── Home / Landing      : http://localhost:{frontend_port}/index.html
     ├── Investor Dashboard  : http://localhost:{frontend_port}/investor.html
     ├── Distributor Portal  : http://localhost:{frontend_port}/distributor.html
     └── AI Showcase (Deep L): http://localhost:{frontend_port}/ai-showcase.html

  ⚙️  Backend API     : http://localhost:{backend_port}
     ├── Health Check        : http://localhost:{backend_port}/health
     ├── Forecaster Endpoint : http://localhost:{backend_port}/api/forecast
     └── Interactive Docs    : http://localhost:{backend_port}/docs

========================================================================
  Press Ctrl+C to stop both servers.
========================================================================
"""
    print(banner, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run Lume AI frontend and backend together")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port (default: 3000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--force", action="store_true", help="Force kill existing processes on ports")
    args = parser.parse_args()

    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing_py_path}".rstrip(":")
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mplconfig")

    # Check port conflicts
    for port, name in [(args.backend_port, "Backend"), (args.frontend_port, "Frontend")]:
        if is_port_in_use(port):
            if args.force:
                print(f"⚠️  Port {port} ({name}) is in use. Killing existing process...", flush=True)
                kill_process_on_port(port)
            else:
                print(f"⚠️  Port {port} ({name}) is already in use.", flush=True)
                print(f"   If an old instance is running, pass --force to kill it: python3 run.py --force", flush=True)

    procs = []

    def cleanup(signum=None, frame=None):
        print("\n🛑 Shutting down Lume AI services...", flush=True)
        for p, name in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(0.5)
        for p, name in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print("✅ All services stopped.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("🚀 Starting Backend (FastAPI / Uvicorn)...", flush=True)
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main_enhanced:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT), env=env)
    procs.append((backend_proc, "Backend"))

    print("🚀 Starting Frontend (HTTP Server on port 3000)...", flush=True)
    frontend_cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(args.frontend_port),
        "--directory",
        str(FRONTEND_DIR),
    ]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(ROOT), env=env)
    procs.append((frontend_proc, "Frontend"))

    # Give them a moment to bind and print banner
    time.sleep(1.0)
    print_banner(args.backend_port, args.frontend_port)

    # Monitor processes
    try:
        while True:
            for p, name in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"❌ {name} process exited unexpectedly with code {ret}", flush=True)
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

