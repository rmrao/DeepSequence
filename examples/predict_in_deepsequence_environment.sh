#!/bin/bash
source ~/miniconda/bin/activate deepsequence
python predict_single_mutant.py ${1} ${2}
conda deactivate
