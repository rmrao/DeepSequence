#!/usr/bin/env python
import sys
import docker
import logging
from docker import types
from pathlib import Path
import argparse
import signal


level = logging.INFO
root_logger = logging.getLogger()
root_logger.setLevel(level)
formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s]   %(message)s", datefmt="%y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
logger = logging.getLogger(__name__)

"""
Args:
    Alignment:
        infile (fasta)
        outfile (a3m)
        seqdb (uniref)
        viral (bool)
        keep_insertions (bool)
    DeepSequence:
        TODO
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile", type=Path, required=True)
parser.add_argument("-o", "--outdir", type=Path, required=True)
parser.add_argument("-n", "--cpu", type=int, default=8)
parser.add_argument("-d", "--seqdb", type=Path, required=True)
parser.add_argument("--viral", action="store_true")
parser.add_argument("--keep_insertions", action="store_true")
args = parser.parse_args()

DOCKER_IMAGE_NAME = "deepsequence"
ROOT_MOUNT_DIRECTORY = Path("/mnt")


command_args = []
mounts = []


def create_mount(mount_name: str, path: Path, read_only: bool = True) -> Path:
    global mounts
    path = path.absolute()
    target_path = ROOT_MOUNT_DIRECTORY / mount_name

    if path.is_file():
        source_path = path.parent
        target_file_path = target_path / path.name
    else:
        source_path = path
        target_file_path = target_path

    logger.info(f"Mounting {source_path} -> {target_path}")
    mount = types.Mount(
        str(target_path), str(source_path), type="bind", read_only=read_only
    )
    mounts.append(mount)
    return target_file_path


infile_path = create_mount("inputs", args.infile)
outdir_path = create_mount("outputs", args.outdir, read_only=False)
db_path = create_mount("uniref100", args.seqdb)

command_args = [
    f"--infile={infile_path}",
    f"--outdir={outdir_path}",
    f"--seqdb={db_path}",
    f"--cpu={args.cpu}",
]
if args.viral:
    command_args.append("--viral")
if args.keep_insertions:
    command_args.append("--keep_insertions")

client = docker.from_env()
container = client.containers.run(
    image=DOCKER_IMAGE_NAME,
    command=command_args,
    runtime="nvidia",
    remove=True,
    detach=True,
    mounts=mounts,
    environment={
        "THEANO_FLAGS": "device=cuda,floatX=float32",
    },
)
# Add signal handler to ensure CTRL+C also stops the running container.
signal.signal(signal.SIGINT, lambda unused_sig, unused_frame: container.kill())

for line in container.logs(stream=True):
    print(line.strip().decode("utf-8"))
