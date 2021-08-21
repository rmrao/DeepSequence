#!/bin/bash

# More safety, by turning some bugs into errors.
# Without `errexit` you don’t need ! and can replace
# PIPESTATUS with a simple $?, but I don’t do that.
set -o errexit -o pipefail -o noclobber -o nounset

USAGE="run_all.sh -i <infile> -o <outfile> -d <seqdb> -n <cpu> [--viral, --keep_insertions]"
# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'getopt test failed, make sure getopt is installed in this system.'
    exit 1
fi

OPTIONS=i:o:d:n:
LONGOPTS=infile:,outdir:,seqdb:,cpu:,viral,keep_insertions

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

infile="" outdir="" seqdb="" cpu=8 viral= keep_insertions=
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -i|--infile)
            infile="$2"
            shift 2
            ;;
        -o|--outdir)
            outdir="$2"
            shift 2
            ;;
        -d|--seqdb)
            seqdb="$2"
            shift 2
            ;;
        -n|--cpu)
            cpu="$2"
            shift 2
            ;;
        --viral)
            viral="--viral"
            shift
            ;;
        --keep_insertions)
            keep_insertions="--keep_insertions"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo $USAGE
            exit 3
            ;;
    esac
done

# handle non-option arguments
if [[ $# -ne 0 ]]; then
    echo $USAGE
    exit 4
fi

if [[ -z "$infile" ]] || [[ -z "$outdir" ]] || [[ -z "$seqdb" ]]; then
    echo $USAGE
    exit 5
fi

set -e
############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

alnfile=${outdir}/`basename ${infile} .fasta`.a3m
prefix=${outdir}/`basename ${infile} .fasta`

conda activate alignment
/app/align.py --infile $infile --outfile $alnfile --seqdb $seqdb --cpu $cpu $viral $keep_insertions
conda deactivate

conda activate deepsequence
/app/run_svi.py --infile $alnfile --outdir $outdir $viral
/app/predict_single_mutant.py --model_prefix $prefix --outdir $outdir
conda deactivate
