#!/bin/bash
#BSUB -J train[1-12]
#BSUB -q gpuv100
#BSUB -W 23:00
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -o outs/train_%J_%I.out
#BSUB -e outs/train_%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Initialize Python environment
source ../irishcream/bin/activate


python spike_detection/tune.py --dataset_idx $LSB_JOBINDEX --results spike_detection/results_tuning_stride_800_9.csv  --comment "jobid: $LSB_JOBID" --hp_tune_type "grid" \
                               --classification_model Filter \
                               --n_models 5 \
                               --n_epochs 1500 \
                               --min_n_epochs_list 800 \
                               --learning_rate_list 0.001 0.0001 0.00001\
                               --dropout_list 0\
                               --l2_penalty_list 0 0.001 \
                               --init_stride_list 2 -1\
                               --depth 9

# run with: bsub < spike_detection/all_dataset_hp_tune_job_array.sh
