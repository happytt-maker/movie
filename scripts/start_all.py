#!/usr/bin/env python
"""Start the MovieLens API and Vue development server together.

Run from any directory with:
    python scripts/start_all.py

Use --frontend-only to preview the checked-in static demo without model artifacts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_API_FILES = (
    Path("models/bpr_mf.pt"),
    Path("data/processed/candidates_test.npz"),
    Path("data/processed/vocab.npz"),
    Path("data/processed/train.npz"),
    Path("data/features/valid.npz"),
)


def ensure_project_venv() -> int | None:
    """Re-run in the project virtual environment when it exists."""
    if os.environ.get("MOVIE_START_ALL_IN_VENV") == "1":
        return None
    if os.name == "nt":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return None
    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except OSError:
        return None
    if current == target:
        return None
    env = os.environ.copy()
    env["MOVIE_START_ALL_IN_VENV"] = "1"
    result = subprocess.run(
        [str(target), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return result.returncode


def prepare_runtime_dirs() -> None:
    for relative in ("data/features", "data/processed", "data/spark", "results"):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def missing_api_files() -> list[Path]:
    return [PROJECT_ROOT / item for item in REQUIRED_API_FILES
            if not (PROJECT_ROOT / item).is_file()]


def resolve_npm() -> str:
    command = "npm.cmd" if os.name == "nt" else "npm"
    resolved = shutil.which(command)
    if not resolved:
        raise RuntimeError(
            f"找不到 {command}。请先安装 Node.js 18 或更高版本，并重新打开终端。"
        )
    return resolved


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    reexec_status = ensure_project_venv()
    if reexec_status is not None:
        return reexec_status

    parser = argparse.ArgumentParser(description="启动电影推荐 API 和前端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8001)
    parser.add_argument("--web-port", type=int, default=5173)
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="只启动前端静态演示，不要求模型和处理后数据",
    )
    args = parser.parse_args()

    prepare_runtime_dirs()
    web_dir = PROJECT_ROOT / "web"
    if not (web_dir / "package.json").is_file():
        print(f"前端目录不完整: {web_dir}", file=sys.stderr)
        return 2

    try:
        npm = resolve_npm()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    commands: list[tuple[str, list[str], Path, dict[str, str]]] = []
    web_env = os.environ.copy()
    web_env["VITE_PROXY_TARGET"] = f"http://{args.host}:{args.api_port}"
    web_env["VITE_DATA_SOURCE"] = "static" if args.frontend_only else "api"

    if not args.frontend_only:
        absent = missing_api_files()
        if absent:
            print("API 所需模型或处理数据尚未生成：", file=sys.stderr)
            for path in absent:
                print(f"  - {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
            print(
                "\n先准备 MovieLens 数据并生成处理文件、BPR 权重和 valid 特征。"
                "\n静态前端可先用：python scripts/start_all.py --frontend-only",
                file=sys.stderr,
            )
            return 2
        python = sys.executable
        api_env = os.environ.copy()
        commands.append((
            "API",
            [python, "-m", "uvicorn", "app.main:app",
             "--host", args.host, "--port", str(args.api_port)],
            PROJECT_ROOT,
            api_env,
        ))

    commands.append((
        "前端",
        [npm, "run", "dev", "--", "--host", args.host,
         "--port", str(args.web_port), "--strictPort"],
        web_dir,
        web_env,
    ))

    children: list[subprocess.Popen] = []
    try:
        for label, command, cwd, env in commands:
            print(f"启动{label}: {' '.join(command)}", flush=True)
            children.append(subprocess.Popen(command, cwd=cwd, env=env))
            time.sleep(1)
            failed = next((p for p in children if p.poll() is not None), None)
            if failed is not None:
                return failed.returncode or 1

        if args.frontend_only:
            print(f"前端已启动: http://{args.host}:{args.web_port}", flush=True)
        else:
            print(f"API 文档: http://{args.host}:{args.api_port}/docs", flush=True)
            print(f"前端页面: http://{args.host}:{args.web_port}", flush=True)
        print("按 Ctrl+C 停止所有服务。", flush=True)

        while True:
            for process in children:
                code = process.poll()
                if code is not None:
                    return code or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在停止服务…", flush=True)
        return 0
    finally:
        for process in reversed(children):
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
