#!/usr/bin/env python3
"""Cynapse Synapse wrapper for MakazhanAlpamys/Soup (soup-cli)."""

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VENV_SOUP = os.path.join(SCRIPT_DIR, ".venv", "bin", "soup")
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")


def soup_bin() -> str:
    return VENV_SOUP if os.path.exists(VENV_SOUP) else "soup"


def run_soup(cmd_args: list[str], timeout: int = 300) -> dict:
    cmd = [soup_bin(), "--no-telemetry"] + cmd_args
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "SUCCESS" if proc.returncode == 0 else "ERROR",
            "returncode": proc.returncode,
            "cmd": " ".join(cmd),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {
            "status": "ERROR",
            "error": f"soup CLI not found at {VENV_SOUP}. Install via: python3 -m venv synapses/soup/.venv && synapses/soup/.venv/bin/pip install 'soup-cli[train]'",
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "cmd": " ".join(cmd)}


def handle_advise(args: dict) -> None:
    dataset = args.get("dataset", "data/cyber_rescue_sft.jsonl")
    res = run_soup(["advise", "run", dataset])
    print(json.dumps(res, indent=2))


def handle_profile(args: dict) -> None:
    model = args.get("model", "LiquidAI/LFM3-1.7B")
    config = args.get("config")
    cmd = ["profile", "--json"]
    if config:
        cmd.extend(["--config", config])
    else:
        cmd.extend(["--model", model])
    res = run_soup(cmd)
    print(json.dumps(res, indent=2))


def handle_data_inspect(args: dict) -> None:
    path = args.get("path", "data/cyber_rescue_sft.jsonl")
    rows = str(args.get("rows", 5))
    res = run_soup(["data", "inspect", path, "--rows", rows])
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            res["line_count"] = sum(1 for line in f if line.strip())
    print(json.dumps(res, indent=2))


def handle_data_doctor(args: dict) -> None:
    path = args.get("path")
    if path:
        res = run_soup(["data", "validate", path])
    else:
        res = run_soup(["doctor"])
    print(json.dumps(res, indent=2))


def handle_train(args: dict) -> None:
    config = args.get("config")
    if config and os.path.exists(config):
        res = run_soup(["train", "--config", config, "--yes"], timeout=3600)
        print(json.dumps(res, indent=2))
        return

    # Fallback: write a minimal soup.yaml in output_dir and launch soup train
    model = args.get("base") or args.get("model") or "LiquidAI/LFM3-1.7B"
    data = args.get("data") or "data/cyber_rescue_sft.jsonl"
    output_dir = args.get("output_dir") or "output/cyber_rescue_lfm3"
    epochs = int(args.get("epochs", 1))
    max_steps = args.get("max_steps")

    os.makedirs(output_dir, exist_ok=True)
    cfg_path = os.path.join(output_dir, "soup.yaml")
    lines = [
        f"base: {model}",
        "task: sft",
        "data:",
        f"  train: {data}",
        "  format: chatml",
        "  max_length: 1024",
        "training:",
        f"  epochs: {epochs}",
        "  batch_size: 1",
        "  gradient_accumulation_steps: 4",
        "  lr: 2e-4",
        "  lora:",
        "    r: 8",
        "    alpha: 16",
        "  quantization: none",
        f"output: {output_dir}",
    ]
    if max_steps is not None:
        lines.insert(lines.index("  batch_size: 1"), f"  max_steps: {int(max_steps)}")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    res = run_soup(["train", "--config", cfg_path, "--yes"], timeout=3600)
    res["config_path"] = cfg_path
    print(json.dumps(res, indent=2))


def handle_export(args: dict) -> None:
    model = args.get("model")
    if not model:
        print(json.dumps({"status": "ERROR", "error": "Missing required 'model' path"}))
        return
    fmt = args.get("format", "gguf")
    quant = args.get("quant", "q4_k_m")
    cmd = ["export", "-m", model, "-f", fmt, "-q", quant]
    if args.get("output"):
        cmd.extend(["-o", args["output"]])
    if args.get("base"):
        cmd.extend(["-b", args["base"]])
    vendored_llama_cpp = os.path.join(REPO_ROOT, "engine", "leafcutter_core", "rust", "llama.cpp")
    if os.path.isdir(vendored_llama_cpp):
        cmd.extend(["--llama-cpp", vendored_llama_cpp])
    res = run_soup(cmd, timeout=1800)
    print(json.dumps(res, indent=2))


def handle_synth(args: dict) -> None:
    mode = args.get("mode", "cyber_rescue")
    py = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    if mode == "cyber_rescue":
        script = os.path.join(SCRIPT_DIR, "prepare_dataset.py")
        out_path = os.path.join(REPO_ROOT, "data", "cyber_rescue_sft.jsonl")
    else:
        script = os.path.join(SCRIPT_DIR, "generate_cynapse_tools.py")
        out_path = args.get("out", os.path.join(REPO_ROOT, "data", "train.jsonl"))

    proc = subprocess.run([py, script], cwd=REPO_ROOT, capture_output=True, text=True)
    rows = 0
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            rows = sum(1 for line in f if line.strip())
    print(json.dumps({
        "status": "SUCCESS" if proc.returncode == 0 else "ERROR",
        "action": "synth",
        "mode": mode,
        "output_file": out_path,
        "sample_count": rows,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }, indent=2))


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No action specified"}))
        sys.exit(1)

    action = sys.argv[1]
    try:
        raw_input = sys.stdin.read()
        args = json.loads(raw_input) if raw_input.strip() else {}
    except Exception:
        args = {}

    handlers = {
        "advise": handle_advise,
        "profile": handle_profile,
        "data_inspect": handle_data_inspect,
        "data_doctor": handle_data_doctor,
        "train": handle_train,
        "export": handle_export,
        "synth": handle_synth,
    }
    handler = handlers.get(action)
    if handler:
        handler(args)
    else:
        print(json.dumps({"status": "ERROR", "error": f"Unknown soup action: {action}"}))


if __name__ == "__main__":
    main()
