#!/bin/bash
#BSUB -J dataset[1-3]
#BSUB -q hpc
#BSUB -W 10
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]" 
#BSUB -R "rusage[mem=2GB]"
#BSUB -o outs/dataset_%J.out
#BSUB -e outs/dataset_%J.err

# Initialize Python environment
source ../irishcream/bin/activate

python spike_detection/generate_datasets.py $LSB_JOBINDEX

# run with: bsub < spike_detection/all_dataset_job_array.sh