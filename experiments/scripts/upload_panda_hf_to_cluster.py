from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

from convert_downloaded_panda_to_hf import output_is_complete
from panda_hf_manifest import CHECKPOINTS, selected, source_candidates


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload converted Panda HF checkpoints to KIT.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--host", default="ka_ugqju@uc3-login2.scc.kit.edu")
    parser.add_argument(
        "--remote-root",
        default="/pfs/work9/workspace/scratch/ka_ugqju-molmoact2-checkpoints/hf_checkpoints",
    )
    parser.add_argument("--socket", required=True)
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
    parser.add_argument("--delete-raw", action="store_true")
    args = parser.parse_args()

    for index, entry in selected(args.start, args.end):
        label = f"{entry['task']}/{entry['mode']}/step{entry['step']}"
        local = args.hf_root / entry["output"]
        remote = f"{args.remote_root}/{entry['output']}"
        print(f"\n[{index}/{len(CHECKPOINTS) - 1}] {label}", flush=True)
        if not output_is_complete(local):
            print("No verified local HF checkpoint; skipping.", flush=True)
            continue
        run(["ssh", "-S", args.socket, args.host, "mkdir", "-p", remote])
        run(
            [
                "rsync",
                "-aP",
                "--partial",
                "-e",
                f"ssh -S {args.socket}",
                f"{local}/",
                f"{args.host}:{remote}/",
            ]
        )
        run(["ssh", "-S", args.socket, args.host, "touch", f"{remote}/upload_complete"])
        print("Uploaded and marked complete.", flush=True)

        if args.delete_raw:
            for relative in source_candidates(entry):
                source = (args.raw_root / relative).resolve()
                raw_root = args.raw_root.resolve()
                if source.is_dir() and source.is_relative_to(raw_root):
                    shutil.rmtree(source)
                    print(f"Removed temporary raw checkpoint: {source}", flush=True)


if __name__ == "__main__":
    main()
