from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from panda_hf_manifest import CHECKPOINTS, selected, source_candidates


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Panda checkpoints from KIT.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--host", default="ka_ugqju@uc3-login2.scc.kit.edu")
    parser.add_argument(
        "--remote-root",
        default="/pfs/work9/workspace/scratch/ka_ugqju-molmoact2-checkpoints/checkpoints",
    )
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--raw-root",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/molmoact2-checkpoints/raw"),
    )
    parser.add_argument("--method", choices=("scp", "rsync"), default="scp")
    args = parser.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for index, entry in selected(args.start, args.end):
        label = f"{entry['task']}/{entry['mode']}/step{entry['step']}"
        print(f"\n[{index}/{len(CHECKPOINTS) - 1}] {label}", flush=True)
        resolved = None
        for relative in source_candidates(entry):
            result = subprocess.run(
                ["ssh", "-S", args.socket, args.host, "test", "-d", f"{args.remote_root}/{relative}"],
                check=False,
                timeout=30,
            )
            if result.returncode == 0:
                resolved = relative
                break
        if resolved is None:
            print(f"MISSING: {label}", flush=True)
            missing.append(label)
            continue

        destination = args.raw_root / resolved
        marker = destination / ".download_complete.json"
        if marker.is_file():
            print("Already downloaded, skipping.", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{args.host}:{args.remote_root}/{resolved}"
        if args.method == "scp":
            run(["scp", "-o", f"ControlPath={args.socket}", "-r", remote, str(destination.parent)])
        else:
            destination.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "rsync",
                    "-aP",
                    "--partial",
                    "--append-verify",
                    "-e",
                    f"ssh -S {args.socket}",
                    f"{remote}/",
                    f"{destination}/",
                ]
            )
        destination.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"index": index, "source": resolved}, indent=2) + "\n")
        print(f"Downloaded: {destination}", flush=True)

    print(f"\nFinished download range. Missing: {len(missing)}")
    for label in missing:
        print(f"  {label}")


if __name__ == "__main__":
    main()
