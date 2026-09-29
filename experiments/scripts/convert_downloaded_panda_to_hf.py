from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

from panda_hf_manifest import CHECKPOINTS, selected, source_candidates


def output_is_complete(output: pathlib.Path) -> bool:
    required = ("config.json", "processor_config.json", "conversion_complete.json")
    if not all((output / name).is_file() for name in required):
        return False
    single = output / "model.safetensors"
    if single.is_file() and single.stat().st_size:
        return True
    index = output / "model.safetensors.index.json"
    if not index.is_file():
        return False
    shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
    return bool(shards) and all((output / shard).is_file() and (output / shard).stat().st_size for shard in shards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded Panda checkpoints to HF format.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--raw-root",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/molmoact2-checkpoints/raw"),
    )
    parser.add_argument(
        "--hf-root",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/molmoact2-checkpoints/hf"),
    )
    args = parser.parse_args()
    args.hf_root.mkdir(parents=True, exist_ok=True)
    log_root = args.hf_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    for index, entry in selected(args.start, args.end):
        label = f"{entry['task']}/{entry['mode']}/step{entry['step']}"
        output = args.hf_root / entry["output"]
        print(f"\n[{index}/{len(CHECKPOINTS) - 1}] {label}", flush=True)
        if output_is_complete(output):
            print("Already converted, skipping.", flush=True)
            continue

        source = next(
            (
                args.raw_root / relative
                for relative in source_candidates(entry)
                if (args.raw_root / relative / ".download_complete.json").is_file()
            ),
            None,
        )
        if source is None:
            print("Not downloaded; skipping.", flush=True)
            continue
        if output.exists():
            backup = output.with_name(f"{output.name}.incomplete-{time.strftime('%Y%m%d-%H%M%S')}")
            output.rename(backup)
            print(f"Preserved incomplete output: {backup}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_root / f"{entry['task']}-{entry['mode']}-step{entry['step']}.log"
        command = [
            "/usr/bin/time",
            "-v",
            sys.executable,
            "-m",
            "olmo.hf_model.convert_molmoact2_to_hf",
            str(source),
            str(output),
            "--attn_implementation",
            "sdpa",
            "--max_shard_size",
            "5GB",
            "--low_memory",
        ]
        print("+", " ".join(command), flush=True)
        with log_file.open("a") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            if process.wait() != 0:
                raise SystemExit(f"Conversion failed: {label}")
        marker = {
            "index": index,
            "task": entry["task"],
            "mode": entry["mode"],
            "step": entry["step"],
            "source_checkpoint": str(source),
            "output_directory": str(output),
            "conversion_mode": "low_memory",
        }
        (output / "conversion_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
        if not output_is_complete(output):
            raise SystemExit(f"HF verification failed: {label}")
        print(f"Converted: {output}", flush=True)


if __name__ == "__main__":
    main()
