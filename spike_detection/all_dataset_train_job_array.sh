#!/bin/bash
#BSUB -J train[1-12]
#BSUB -q gpuv100
#BSUB -W 5:00
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -o outs/train_%J_%I.out
#BSUB -e outs/train_%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Initialize Python environment
source ../irishcream/bin/activate

python spike_detection/train.py --models AveragePrediction LinearRegression ThresholdBased FreshPRINCERegressor DrCIFRegressor InceptionTme --dataset_idx $LSB_JOBINDEX --results spike_detection/results.csv --save_model_path models --comment "jobid: $LSB_JOBID"
#python spike_detection/train.py --models LinearRegression ThresholdBased FreshPRINCERegressor DrCIFRegressor InceptionTime --dataset_idx $LSB_JOBINDEX --results spike_detection/results.csv --save_model_path models --comment "jobid: $LSB_JOBID"


# run with: bsub < spike_detection/all_dataset_train_job_array.sh
