
for IDX in 2 4 6 8 10 12
do
echo "dataset idx $IDX"
python spike_detection/train.py --models AveragePrediction LinearRegression ThresholdBased DrCIFRegressor \
                                --dataset_idx $IDX --results spike_detection/results.csv \
                                --comment "batch" --init_stride 2
done
