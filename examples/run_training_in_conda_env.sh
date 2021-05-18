#!/bin/bash
source ~/miniconda/bin/activate deepsequence
if [[ $# -eq 2 ]]
then
    python run_svi.py ${1} ${2}
elif [[ $# -eq 3 ]]
then
    python run_svi.py ${1} ${2} --ensemble ${3}
else
    echo "Invalid number of arguments" >&2
fi
conda deactivate
