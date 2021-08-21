#!/usr/bin/env python2
import numpy as np
import time
import sys
import cPickle
sys.path.insert(0, "/app/DeepSequence/")
import model
import helper
import train
import os

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   False,
    }


def train_model(alignment_file, output_dir, ensemble=None, viral=False):
    name = os.path.basename(alignment_file).rsplit(".", 1)[0]
    param_file = os.path.join(output_dir, name + "_params.pkl")
    if os.path.exists(param_file):
        print("Param file already exists, skipping.")
        return

    if train_params["verbose"]:
        print("Starting training")
    if ensemble is not None:
        model_params["r_seed"] += ensemble + 1

    data_helper = helper.DataHelper(alignment_file=alignment_file,
                                    calc_weights=True,
                                    theta=0.01 if viral else 0.2)

    vae_model   = model.VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        )

    job_string = helper.gen_job_string({"filename": os.path.basename(alignment_file).rsplit(".", 1)[0]}, model_params)

    print (job_string)

    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)

    vae_model.save_parameters(file_prefix=name, path=output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ensemble", default=None, type=int)
    parser.add_argument("--viral", action="store_true")
    args = parser.parse_args()
    train_model(args.infile, args.outdir, args.ensemble, args.viral)
