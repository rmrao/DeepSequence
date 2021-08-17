#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description="Runs deepsequence alignment and training.")
parser.add_argument("-i", "--infile", type=Path, required=True)
parser.add_argument("-o", "--outdir", type=Path, required=True)
parser.add_argument("-d", "--seqdb", type=Path, required=True)
parser.add_argument("--viral", action="store_true")
parser.add_argument("--keep_insertions", action="store_true")
parser.add_argument("-n", "--cpu", type=int, default=8)
args = parser.parse_args()

fasta_file = args.infile
aln_file = args.outdir / fasta_file.with_suffix(".a3m").name
prefix = args.outdir / fasta_file.stem

align_command = ["/app/align.py", f"--infile={fasta_file}", f"--outfile={aln_file}", f"--seqdb={args.seqdb}", f"--cpu={args.cpu}"]
train_command = ["/app/run_svi.py", f"--infile={aln_file}", f"--outdir={args.outdir}"]
predict_command = ["/app/predict_single_mutant.py", f"--model_prefix={prefix}", f"--outdir={args.outdir}"]

if args.viral:
    align_command.append("--viral")
    train_command.append("--viral")
if args.keep_insertions:
    align_command.append("--keep_insertions")

subprocess.run(align_command).check_returncode()
subprocess.run(train_command).check_returncode()
subprocess.run(predict_command).check_returncode()
