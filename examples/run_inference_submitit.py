from timeit import default_timer as timer
import submitit
from filelock import FileLock
from pathlib import Path
from typing import List
import subprocess
import os
from functools import partial

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["THEANO_FLAGS"] = "floatX=float32,device=cuda"


def commands(alignment_dir: Path, output_dir: Path):
    for msa in alignment_dir.glob("*.a3m"):
        param_prefix = (
            f"vae_output_encoder-1500-1500_Nlatent-30_decoder-100-500_filename-{msa.stem.split('.')[0]}_bs-100_conv_pat"
            "-True_d_c_size-40_final_decode_nonlin-sigmoid_final_pwm_scale-True_logit_p-0.001_n_pat-4_r_seed-1234_"
            "sparsity-logit"
        )
        param_file = output_dir / param_prefix
        yield ["bash", "predict_in_deepsequence_environment.sh", str(msa), str(param_file)]


def timed_run(command: List[str], output_dir: Path) -> None:
    start = timer()
    subprocess.run(command).check_returncode()
    tottime = timer() - start
    timing_results = output_dir / "timing_results_inference.csv"
    lockfile = timing_results.with_name(timing_results.name + ".lock")
    with FileLock(lockfile):
        with open(timing_results, "a") as f:
            f.write(f"{command[2]},{tottime}\n")


def main(args):

    files = Path(args.alignment_dir).glob("*.a3m")
    output_dir = Path(args.output_dir)

    def commands():
        for file in files:
            base_command = [
                "bash",
                "run_training_in_conda_env.sh",
                str(file),
                str(output_dir),
            ]
            yield base_command

    executor = submitit.AutoExecutor(folder=f"/checkpoint/{os.environ['USER']}/deepsequence-timing-logs")
    executor.update_parameters(
        timeout_min=3000,
        slurm_partition="learnfair",
        gpus_per_node=1,
        mem_gb=64,
        cpus_per_task=10,
        slurm_constraint="volta32gb",
        slurm_array_parallelism=32,
    )

    runfunc = partial(timed_run, output_dir=output_dir)
    with executor.batch():
        for command in commands:
            executor.submit(runfunc, command)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment_dir", type=str, help="directory containing alignment files")
    parser.add_argument("output_dir", type=str, help="directory in which to save parameters, timing results")
    args = parser.parse_args()
    main(args)
