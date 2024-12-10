#!/bin/bash
#BSUB -J train[1-12]
#BSUB -q gpuv100
#BSUB -W 10:00
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -o outs/train_%J_%I.out
#BSUB -e outs/train_%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Initialize Python environment
source ../irishcream/bin/activate


python spike_detection/tune.py --dataset_idx $LSB_JOBINDEX --results spike_detection/results_tuning_stride.csv  --comment "jobid: $LSB_JOBID" --hp_tune_type "grid" \
                               --learning_rate_list 0.0001\
                               --dropout_list 0.2 \
                               --l2_penalty_list 0.0001 \
                               --init_stride_list 3 -1 2


# run with: bsub < spike_detection/all_dataset_hp_tune_job_array.sh
