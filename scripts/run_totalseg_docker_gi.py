#!/usr/bin/python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys


DEFAULT_DOCKER = "/Applications/Docker.app/Contents/Resources/bin/docker"
DEFAULT_HOST = f"unix://{Path.home()}/.docker/run/docker.sock"
DEFAULT_IMAGE = "wasserth/totalsegmentator:2.14.0"
DEFAULT_PLATFORM = "linux/amd64"
DEFAULT_MIN_FREE_GB = 35.0


def _docker_bin() -> str | None:
    configured = os.environ.get("DOCKER_BIN")
    if configured:
        return configured
    if Path(DEFAULT_DOCKER).exists():
        return DEFAULT_DOCKER
    return shutil.which("docker")


def _socket_ping(docker_host: str) -> None:
    if not docker_host.startswith("unix://"):
        return
    socket_path = docker_host.removeprefix("unix://")
    if not Path(socket_path).exists():
        raise RuntimeError(f"Docker socket not found: {socket_path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5.0)
        client.connect(socket_path)
        client.sendall(b"GET /_ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        response = client.recv(4096)
    if b"OK" not in response:
        raise RuntimeError(f"Docker daemon did not answer _ping on {socket_path}")


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024 / 1024 / 1024


def _build_command(args: argparse.Namespace, docker_bin: str, docker_host: str) -> list[str]:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    image = os.environ.get("TOTALSEG_DOCKER_IMAGE", DEFAULT_IMAGE)
    platform = os.environ.get("TOTALSEG_PLATFORM", DEFAULT_PLATFORM)
    return [
        docker_bin,
        "--host",
        docker_host,
        "run",
        "--rm",
        "--platform",
        platform,
        "-v",
        f"{input_path.parent}:/input:ro",
        "-v",
        f"{output_dir}:/output",
        image,
        "TotalSegmentator",
        "-i",
        f"/input/{input_path.name}",
        "-o",
        "/output",
        *args.segmenter_args,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TotalSegmentator GI segmentation through Docker Desktop.")
    parser.add_argument("-i", "--input", required=True, help="Input CT NIfTI.")
    parser.add_argument("-o", "--output", required=True, help="Output mask directory.")
    parser.add_argument("--plan-only", action="store_true", help="Print the Docker command without running it.")
    args, extra = parser.parse_known_args(argv)
    args.segmenter_args = extra

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        print(f"Input CT not found: {input_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    docker_bin = _docker_bin()
    if docker_bin is None:
        print("Docker CLI not found. Start Docker Desktop or set DOCKER_BIN.", file=sys.stderr)
        return 2

    docker_host = os.environ.get("DOCKER_HOST", DEFAULT_HOST)
    try:
        _socket_ping(docker_host)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    min_free_gb = float(os.environ.get("TOTALSEG_MIN_FREE_GB", str(DEFAULT_MIN_FREE_GB)))
    free_gb = _disk_free_gb(output_dir)
    low_disk = free_gb < min_free_gb and os.environ.get("TOTALSEG_ALLOW_LOW_DISK") != "1"

    command = _build_command(args, docker_bin, docker_host)
    print("Planned TotalSegmentator Docker command:")
    print("  " + " ".join(shlex.quote(part) for part in command))
    print(f"Free disk at output path: {free_gb:.1f} GiB")

    if low_disk:
        print(
            f"Refusing to start TotalSegmentator Docker run: only {free_gb:.1f} GiB free, "
            f"guard requires {min_free_gb:.1f} GiB.",
            file=sys.stderr,
        )
        print(
            "The Docker image plus unpacked layers/model cache need substantially more room.",
            file=sys.stderr,
        )
        if not args.plan_only and os.environ.get("TOTALSEG_PLAN_ONLY") != "1":
            return 3

    if args.plan_only or os.environ.get("TOTALSEG_PLAN_ONLY") == "1":
        return 0

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
