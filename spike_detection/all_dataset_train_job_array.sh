#!/bin/bash
#BSUB -J train[1-5]
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]" 
#BSUB -R "rusage[mem=2GB]"
#BSUB -o outs/train_%J.out
#BSUB -e outs/train_%J.err

# Initialize Python environment
source ../irishcream/bin/activate

#python spike_detection/train.py --models InceptionNet --dataset_idx $LSB_JOBINDEX --results spike_detection/results.csv --save_model_path models
python spike_detection/train.py --models LinearRegression FreshPRINCE InceptionNet --results spike_detection/results.csv --save_model_path models


# run with: bsub < spike_detection/all_dataset_train_job_array.sh