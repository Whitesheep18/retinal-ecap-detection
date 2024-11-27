# retinal-ecap-detection
Master Thesis of Marie Amalie Adergaard Lunde (s194637) & Anna Reisz (s220817)
Title: Data-Driven Analysis of Retinal Response Patterns to Carbon Electrode Stimulation

## Abstract
[insert abstract from thesis]

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

<span style="font-size: 8px;">*Irish cream is the suggested type of tea for this exercise</span>


## Acknowledgements
The recordings used as the foundation for synthetic data generation were provided courtesy of Aarhus University Hospital. For more information, please contact Jesper Gudsmed Madsen. (jesper.madsen@clin.au.dk)

## References
Kusumanchi, Pratik et al. (2024). “Electrical stimulation of neuroretinas with 3D pyrolytic carbon
electrodes.” In: ASC Biomaterials Science Engineering. Manuscript submitted for publication.