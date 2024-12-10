#!/bin/bash
#BSUB -J gen[1-12]
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]" 
#BSUB -R "rusage[mem=2GB]"
#BSUB -o outs/dataset_%J.out
#BSUB -e outs/dataset_%J.err
#BSUB -Ne

# Initialize Python environment
source ../irishcream/bin/activate

python src/generate_datasets.py $LSB_JOBINDEX 

# run with: bsub < src/all_dataset_job_array.sh