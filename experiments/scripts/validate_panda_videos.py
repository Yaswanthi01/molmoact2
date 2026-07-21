#!/usr/bin/env python3
"""Exhaustively validate every video in the converted Panda drawer dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import av


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "panda_drawer_ee_molmoact2_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Converted LeRobot dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    videos_link = dataset_root / "videos"

    if not info_path.is_file():
        raise SystemExit(f"Missing dataset metadata: {info_path}")
    if not videos_link.exists():
        raise SystemExit(f"Missing videos directory: {videos_link}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_total_frames = int(info["total_frames"])
    expected_fps = float(info["fps"])
    camera_specs = {
        key: feature
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    }

    videos_root = videos_link.resolve()
    video_paths = sorted(videos_root.rglob("*.mp4"))
    if not video_paths:
        raise SystemExit(f"No MP4 files found under {videos_root}")

    paths_by_camera: dict[str, list[Path]] = defaultdict(list)
    for path in video_paths:
        relative = path.relative_to(videos_root)
        paths_by_camera[relative.parts[0]].append(path)

    assert set(paths_by_camera) == set(camera_specs), (
        f"Camera directories do not match metadata: "
        f"videos={sorted(paths_by_camera)}, metadata={sorted(camera_specs)}"
    )

    print(f"Dataset: {dataset_root}")
    print(f"Physical videos directory: {videos_root}")
    print(f"Video files: {len(video_paths)}")

    errors: list[str] = []
    for camera_key, paths in sorted(paths_by_camera.items()):
        spec = camera_specs[camera_key]["info"]
        expected_width = int(spec["video.width"])
        expected_height = int(spec["video.height"])
        expected_codec = str(spec["video.codec"])
        camera_frame_count = 0

        print(f"\n{camera_key}: {len(paths)} files")
        for file_number, path in enumerate(paths, start=1):
            relative = path.relative_to(videos_root)
            decoded_frames = 0
            try:
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError("no video stream")
                    stream = container.streams.video[0]
                    codec = stream.codec_context.name
                    if codec == "libdav1d":
                        codec = "av1"
                    fps = float(stream.average_rate) if stream.average_rate else None

                    if stream.width != expected_width or stream.height != expected_height:
                        errors.append(
                            f"{relative}: resolution {stream.width}x{stream.height}, "
                            f"expected {expected_width}x{expected_height}"
                        )
                    if codec != expected_codec:
                        errors.append(f"{relative}: codec {codec}, expected {expected_codec}")
                    if fps is None or abs(fps - expected_fps) > 0.01:
                        errors.append(f"{relative}: FPS {fps}, expected {expected_fps}")

                    for frame in container.decode(stream):
                        if frame.width != expected_width or frame.height != expected_height:
                            raise ValueError(
                                f"decoded frame has resolution {frame.width}x{frame.height}"
                            )
                        decoded_frames += 1

                if decoded_frames == 0:
                    errors.append(f"{relative}: decoded zero frames")
                camera_frame_count += decoded_frames
                print(
                    f"  [{file_number}/{len(paths)}] {relative}: "
                    f"{decoded_frames} frames"
                )
            except Exception as exc:
                errors.append(f"{relative}: decode failed: {exc}")
                print(f"  [{file_number}/{len(paths)}] {relative}: FAILED")

        if camera_frame_count != expected_total_frames:
            errors.append(
                f"{camera_key}: decoded {camera_frame_count} total frames, "
                f"expected {expected_total_frames}"
            )
        print(
            f"  Total: {camera_frame_count} frames "
            f"(expected {expected_total_frames})"
        )

    if errors:
        print("\nVIDEO VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nFULL VIDEO VALIDATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVideo validation interrupted.", file=sys.stderr)
        raise SystemExit(130)
