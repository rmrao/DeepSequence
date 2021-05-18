from __future__ import print_function
import sys
import os
sys.path.insert(0, "../DeepSequence/")
import model  # noqa: E402
import helper  # noqa: E402

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment_file")
    parser.add_argument("file_prefix")
    args = parser.parse_args()

    basename = os.path.basename(args.alignment_file)[:-4]
    if not os.path.exists(args.file_prefix + "_params.pkl"):
        raise OSError("File Not Found! " + args.file_prefix)

    data_helper = helper.DataHelper(alignment_file=args.alignment_file,
                                    calc_weights=False)

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
    print("Model built")
    vae_model.load_parameters(file_prefix=args.file_prefix)
    print("Parameters loaded")

    mutant_name_list, delta_elbos = data_helper.single_mutant_matrix(vae_model, N_pred_iterations=2000)
    with open(os.path.join(os.path.dirname(args.file_prefix), "basename" + ".csv"), "w") as f:
        f.write("mutant,score\n")
        for mutant, elbo in zip(mutant_name_list, delta_elbos):
            f.write(mutant + "," + str(elbo) + "\n")
