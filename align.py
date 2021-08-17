#!/usr/bin/env python3
from typing import List, Tuple, Union, Iterator, Sequence
from timeit import default_timer as timer
import math
import sys
from pathlib import Path
import subprocess
import tempfile
import logging
import re
import contextlib
import os

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.spatial.distance import squareform, pdist

PathLike = Union[str, Path]

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


class MSA:
    """Class that represents a multiple sequence alignment."""

    def __init__(
        self,
        sequences: List[Tuple[str, str]],
        seqid_cutoff: float = 0.2,
    ):
        self.headers = [header for header, _ in sequences]
        self.sequences = [seq for _, seq in sequences]
        self._seqlen = len(self.sequences[0])
        self._depth = len(self.sequences)
        self.seqid_cutoff = seqid_cutoff
        assert all(
            len(seq) == self._seqlen for seq in self.sequences
        ), "Seqlen Mismatch!"

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.headers, self.sequences)

    def select(self, indices: Sequence[int], axis: str = "seqs") -> "MSA":
        assert axis in ("seqs", "positions")
        if axis == "seqs":
            data = [(self.headers[idx], self.sequences[idx]) for idx in indices]
            return self.__class__(data)
        else:
            data = [
                (header, "".join(seq[idx] for idx in indices)) for header, seq in self
            ]
            return self.__class__(data)

    def lowercase_columns(self, indices: Sequence[int]) -> "MSA":
        to_lower = set(indices)
        data = [
            (
                header,
                "".join(
                    aa.lower() if i in to_lower else aa for i, aa in enumerate(seq)
                ),
            )
            for header, seq in self
        ]
        return self.__class__(data)

    def filter_coverage(self, threshold: float, axis: str = "seqs") -> "MSA":
        assert 0 <= threshold <= 1
        assert axis in ("seqs", "positions")
        notgap = self.array != self.gap
        match = notgap.mean(1 if axis == "seqs" else 0)
        if axis == "seqs":
            indices = np.where(match >= threshold)[0]
            return self.select(indices, axis=axis)
        else:
            indices = np.where(match < threshold)[0]
            return self.lowercase_columns(indices)

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> "MSA":

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tempdirname, open(
            os.devnull, "w"
        ) as devnull:
            tempdir = Path(tempdirname)
            fasta_file = tempdir / "input.fasta"
            fasta_file.write_text(
                "\n".join(f">{i}\n{seq}" for i, seq in enumerate(self.sequences))
            )
            output_file = tempdir / "output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    "-M a3m",
                    f"-o {output_file}",
                    f"-id {seqid}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            result = subprocess.run(command, stdout=devnull, stderr=devnull)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return self.select(indices, axis="seqs")

    @property
    def gap(self) -> Union[bytes, int]:
        return b"-" if self.dtype == np.dtype("S1") else ord("-")

    @property
    def array(self) -> np.ndarray:
        if not hasattr(self, "_array"):
            self._array = np.array([list(seq) for seq in self.sequences], dtype="|S1")
        return self._array

    @property
    def dtype(self) -> type:
        return self._array.dtype

    @dtype.setter
    def dtype(self, value: type) -> None:
        assert value in (np.uint8, np.dtype("S1"))
        self._array = self._array.view(value)

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def seqid_cutoff(self) -> float:
        return self._seqid_cutoff

    @seqid_cutoff.setter
    def seqid_cutoff(self, value: float) -> None:
        assert 0 <= value <= 1
        if getattr(self, "_seqid_cutoff", None) != value:
            with contextlib.suppress(AttributeError):
                delattr(self, "_weights")
            with contextlib.suppress(AttributeError):
                delattr(self, "_neff")
        self._seqid_cutoff = value

    @property
    def is_covered(self) -> np.ndarray:
        if not hasattr(self, "_is_covered"):
            self._is_covered = (self.array[1:] != self.gap).any(0)
        return self._is_covered

    @property
    def coverage(self) -> float:
        if not hasattr(self, "_coverage"):
            self._coverage = self.is_covered.mean()
        return self._coverage

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = 1 / (
                squareform(pdist(self.array, "hamming")) < self.seqid_cutoff
            ).sum(1)
        return self._weights

    def neff(self, normalization: Union[float, str] = "none") -> float:
        if isinstance(normalization, str):
            assert normalization in ("none", "sqrt", "seqlen")
            normalization = {
                "none": 1,
                "sqrt": math.sqrt(self.seqlen),
                "seqlen": self.seqlen,
            }[normalization]
        if not hasattr(self, "_neff"):
            self._neff = self.weights.sum()
        return self._neff / normalization

    @classmethod
    def from_stockholm(
        cls,
        stofile: PathLike,
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(stofile, "stockholm"):
            description = record.description
            sequence = str(record.seq)
            if not keep_insertions:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa != "-"]
                sequence = "".join(sequence[idx] for idx in valid_indices)
            output.append((description, sequence))
        return cls(output, **kwargs)

    @classmethod
    def from_fasta(
        cls,
        fasfile: PathLike,
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        for record in SeqIO.parse(fasfile, "fasta"):
            description = record.description
            sequence = str(record.seq)
            if not keep_insertions:
                sequence = re.sub(r"([a-z]|\.|\*)", "", sequence)
            output.append((description, sequence))
        return cls(output, **kwargs)

    @classmethod
    def from_file(
        cls,
        alnfile: PathLike,
        keep_insertions: bool = False,
        **kwargs,
    ) -> "MSA":
        filename = Path(alnfile)
        if filename.suffix == ".sto":
            return cls.from_stockholm(filename, keep_insertions, **kwargs)
        elif filename.suffix in (".fas", ".fasta", ".a3m"):
            return cls.from_fasta(filename, keep_insertions, **kwargs)
        else:
            raise ValueError(f"Unknown file format {filename.suffix}")

    def write(self, outfile: PathLike, form: str = "fasta") -> None:
        SeqIO.write(
            (
                SeqIO.SeqRecord(Seq(seq), description=header, id=header)
                for header, seq in self
            ),
            outfile,
            form,
        )


def run_jackhmmer(
    seqfile: PathLike,
    seqdb: PathLike,
    outfile: PathLike,
    bitscore: float,
    num_iter: int = 5,
    num_cpu: int = 8,
    jackhmmer_bin: str = "jackhmmer",
):

    command = " ".join(
        [
            f"{jackhmmer_bin}",
            "-o /dev/null",
            "--notextw",
            "--noali",
            f"--incT {bitscore}",
            f"--cpu {num_cpu}",
            f"-N {num_iter}",
            f"-A {outfile}",
            f"{seqfile}",
            f"{seqdb}",
        ]
    ).split(" ")
    result = subprocess.run(command)
    result.check_returncode()


def align(
    seqfile: PathLike,
    outfile: PathLike,
    seqdb: PathLike,
    is_viral: bool,
    keep_insertions: bool,
    tempdir: PathLike,
    num_cpu: int = 8,
) -> None:
    """An attempt to match the following text from DeepSequence:

    For each analyzed protein (target sequence), multiple sequence alignments of the
    corresponding protein family were obtained by the default five search iterations of
    the profile HMM homology search tool jackhmmer against the UniRef100 database of
    non-redundant protein sequences (release 11/2015). To control for comparable
    evolutionary depth across different families, we used length-normalized bit scores
    to threshold sequence similarity rather than E-values. A default bit score of 0.5
    bits/residue was used as a threshold for inclusion unless the align-ment yielded
    <80% coverage of the length of the target domain or if there were not enough
    sequences (redundancy-reduced number of sequences < 10L); in the first case, the
    threshold was increased in steps of 0.05 bits/residue until suf-ficient coverage
    was obtained; in the second case, the threshold was decreased until there were
    sufficient sequences (q10L). If these two objectives were conflicting, precedence
    was given to maintaining more than 10L sequences. Since the sequence diversity of
    viral protein families is typically much lower than that of bacterial and eukaryotic
    families, the alignment depth for viral proteins was chosen as a default of 0.5
    bits/residue even if the redundancy-reduced number of sequences was lower than 10L.
    The alignments were post-processed to exclude positions with more than 30% gaps and
    to exclude sequence fragments that align to less than 50% of the length of the
    target sequence. For the ParE-ParD toxin-antitoxin interaction, a joint sequence
    alignment with matched homologs of both interaction partners was generated using
    our previously described approach EVcomplex. Alignments for RNA sequence families
    were obtained from the Rfam database and redundancy-reduced at the same 80%
    identity cutoff as proteins. The tRNA alignment was filtered to contain only
    sequences with a CCU anticodon.

    """
    outfile = Path(outfile)
    if outfile.exists():
        print("Alignment already exists, skipping.")
        return
    seqfile = Path(seqfile)
    tempdir = Path(tempdir)
    tempoutfile = tempdir / Path(seqfile).with_suffix(".sto").name

    record = SeqIO.read(seqfile, "fasta")
    seqlen = len(record)
    bits_per_residue = 0.5

    bitscores_tried = set()
    start = timer()
    while True:
        if bits_per_residue <= 0.01:
            break
        logger.info(f"Running jackhmmer with {bits_per_residue:0.2f} bits / residue")
        bitscores_tried.add(bits_per_residue)
        run_jackhmmer(seqfile, seqdb, tempoutfile, bits_per_residue * seqlen, num_cpu=num_cpu)
        msa = MSA.from_file(tempoutfile, seqid_cutoff=0.80 if not is_viral else 0.99)
        neff = msa.hhfilter(seqid=80 if not is_viral else 99).depth
        too_few_sequences = neff < 10 * msa.seqlen
        coverage_too_low = msa.coverage < 0.8

        if too_few_sequences and not is_viral:
            logger.info(
                f"{neff} / {10 * msa.seqlen} effective sequences found, "
                "reducing bitscore"
            )
            bits_per_residue -= 0.05
        elif coverage_too_low:
            logger.info(f"Coverage below 80% threshold: {msa.coverage:0.1%}")
            bits_per_residue += 0.05
            if bits_per_residue in bitscores_tried:
                break
        else:
            break

    tottime = timer() - start
    logger.info(f"Finished alignment in {int(tottime)}s")

    msa = msa.filter_coverage(0.5, axis="seqs")
    if keep_insertions:
        records = SeqIO.parse(tempoutfile, "stockholm")
        SeqIO.write(records, outfile, "fasta")
    else:
        msa = msa.filter_coverage(0.3, axis="positions")
        msa.write(outfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=Path, required=True)
    parser.add_argument("-o", "--outfile", type=Path, required=True)
    parser.add_argument("-d", "--seqdb", type=Path, required=True)
    parser.add_argument("--viral", action="store_true")
    parser.add_argument("--keep_insertions", action="store_true")
    parser.add_argument("-n", "--cpu", type=int, default=8)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        align(
            args.infile,
            args.outfile,
            args.seqdb,
            args.viral,
            args.keep_insertions,
            tempdir,
            num_cpu=args.cpu,
        )
