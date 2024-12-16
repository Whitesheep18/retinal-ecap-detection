
for IDX in 1 3 5 7 9 11
do
python spike_detection/train.py --models AveragePrediction LinearRegression ThresholdBased DrCIFRegressor \
                                --dataset_idx $IDX --results spike_detection/results.csv \
                                --comment "batch" --init_stride 2
done
