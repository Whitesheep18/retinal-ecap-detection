#!/bin/bash
#BSUB -J dataset[1-3]
#BSUB -q hpc
#BSUB -W 10
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]" 
#BSUB -R "rusage[mem=2GB]"
#BSUB -o outs/train_%J.out
#BSUB -e outs/train_%J.err

# Initialize Python environment
source ../irishcream/bin/activate

python spike_detection/train.py --models LinearRegression --dataset_idx $LSB_JOBINDEX --results spike_detection/results.csv
# multiple models: python spike_detection/train.py --models LinearRegression FreshPRINCE InceptionNet --dataset_idx $LSB_JOBINDEX


# run with: bsub < spike_detection/all_dataset_job_array.sh