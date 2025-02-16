# retinal-ecap-detection
Master Thesis of Marie Amalie Adergaard Lunde (s194637) & Anna Reisz (s220817)
Title: Data-Driven Analysis of Retinal Response Patterns to Carbon Electrode Stimulation

## Abstract
This study explores how retinal ganglion cells respond to electrical stimuli by identifying and counting
overlapping (compound) action potentials, a critical step in evaluating electrode designs for retinal
implants. Inspired by the work in Kusumanchi et al. 2025, where two distinct electrode designs for
retinal implants are introduced and assessed using experimental data, this study aims to enhance the
counting methods used for performance evaluation. Due to the extreme noise and irregularities in the
experimental data, synthetic datasets were created, designed to emulate the characteristics of the ex-
perimental data, allowing the use of supervised methods. The 12 synthetic datasets with varying noise
levels enabled model evaluation under different conditions, addressing the unknown noise level in the
experimental data.
To count action potentials in retinal ganglion cells, a two-phase solution pipeline is proposed. The
first stage involves classifying whether cellular activity is present or not, while the second estimates the
number of action potentials in the sample using regression. Both baseline and advanced methods are
employed for these tasks. The results demonstrate that advanced models outperform baseline methods
in both classification and regression. For classification, the state-of-the-art model HIVE-COTE-V2 per-
formed significantly better than Random Forest with an F1-score of 0.96 ± 0.01 across all simulated
datasets, albeit with a higher computational cost.
For the regression stage, the study investigated if a modified version of InceptionTimeE designed with
specific alterations to suit the requirements of spike counting better, could outperform off-the-shelf
state-of-the-art methods for time series extrinsic regression. The proposed method was on par with
the advanced models recommended by the literature and performed especially well in low-noise settings
(RMSE 6.23). While it performed well in low-noise conditions (RMSE 5.90), it did not significantly
outperform similar models. As a result, FreshPRINCE, a feature-based rotation forest regressor with
the best overall performance (average RMSE 9.65), was selected for the final solution pipeline.
Lastly, this study applied the solution pipeline (HIVE-COTE-V2 and FreshPRINCE) to the experi-
mental data, where it was observed that the simulated data does not perfectly replicate experimental
conditions. As a consequence, the solution pipeline failed to predict action potentials in line with
expectations. This suggests that either the experimental data’s noise levels should be decreased or
the simulated dataset should better capture the variability in the experimental data to ensure reliable
spike-counting performance in practical applications.

## Repo Structure
Exploratory data analysis is `eda/` and advanced analysis in `ml_scripts`. This is also where you find implementation of the method proposed in (Kusumanchi et al. 2024).

You can find helper functions for the methods in `src/`, such as the dataset simulation procedure in `recording_generator.py`. These are the elements that are used in `spike_detection/` such as training of the various models in `train.py`.

Adaptation of the `IncpetionTime` model for pytorch for time series extrinsic regression can be found in `src/inception_time`.


## Not-so-quick-start
This guide assumes that you are working on DTU's HPC.

Create virtual environment:

```
module load python3/3.10.12
python -m venv ../irishcream
source ../irishcream/bin/activate
pip install -r requirements.txt
```

Create all datasets:
```
bsub < spike_detection/all_dataset_job_array.sh
```
Train all models:
```
bsub < spike_detection/all_dataset_train_job_array.sh
```
You'll find results in `spike_detection/results.csv` which can be visualized using `src/visualize/results.py`

<span style="font-size: 8px;">*Irish cream is the suggested type of tea for this exercise</span>


## Acknowledgements
The recordings used as the foundation for synthetic data generation were provided courtesy of Aarhus University Hospital. For more information, please contact Jesper Gudsmed Madsen. (jesper.madsen@clin.au.dk)

## References
Kusumanchi, Pratik et al. (2024). “Electrical stimulation of neuroretinas with 3D pyrolytic carbon
electrodes.” In: ASC Biomaterials Science Engineering. Manuscript submitted for publication.
