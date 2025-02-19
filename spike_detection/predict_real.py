from src.utils import get_signal_by_type, eye_experiment_sa_start, save_figure, sorting_key
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import datetime as dt
import sys
warnings.filterwarnings("ignore")


colors = {"non-stimulated": "#651b7e",
          "stimulated": "#dd4f66",
          "TTX": "#fca572"}

design = sys.argv[1]
print(f'{design} design')

# for each dataset, get model, predict for all designs, eyes and experiments.
# save results
clf_name = 'HIVECOTEV2'
model_name = 'FreshPRINCERegressor'
results_file = 'real_data_preds.csv'

with open(results_file, 'w') as f:
    f.write('Date,Model,Classification model,Dataset,Design,Eye,Experiment,y_pred\n')

datasets = [x for x in os.listdir('../simulated_data') if x.startswith('DS')]
datasets = sorted(datasets, key=sorting_key)
# reverse sort
datasets = datasets[::-1]

#for dataset in datasets:

def get_label_from_ds(dataset_name):
    _, white, me, _ = dataset_name.split('_')
    return f'{white}      {me} '


sa_length = 300
response_length = 2700
experiment_length = sa_length + response_length
max_stimuli = 120
channel_id = 0

experiment_types = ['stimulated', 'non-stimulated', 'TTX']

fig, axs = plt.subplots(2, 3, constrained_layout=True)
fig.set_figwidth(15)
fig.suptitle(f'{design} design')
for eye in range(1, 7):
    print(f'\t Eye {eye}')
    eye_idx = eye-1
    ax = axs[eye_idx//3, eye_idx%3]
    ax.set_title(f'Eye {eye}')
    ax.set_xlim(40, 85)
    for j, dataset in enumerate(datasets):
        print("\t \t Getting models for datasets", dataset)
        if clf_name is not None:
            clf = pickle.load(open(f'../models/{clf_name}_{dataset}.pkl', 'rb'))
        model = pickle.load(open(f'../models/{model_name}_{dataset}.pkl', 'rb'))
        boxes = []
        for k, experiment in enumerate(experiment_types):
            print(f'\t \t \t Predicting {experiment}')
            time, signal = get_signal_by_type(eye=eye, design=design, experiment=experiment, verbose=0)
            signal = signal[:, channel_id]

            offset = eye_experiment_sa_start[design][eye][experiment]

            X = []
            i = 0
            while offset + (i+1)*experiment_length < len(signal) and i < max_stimuli:
                x = signal[offset + i*experiment_length + sa_length:offset + (i+1)*experiment_length] # am I off by 1?
                X.append(x)
                i += 1
            X = np.array(X)

            if clf_name is not None:
                #print(f'Classifying {len(X)} samples')
                y_class = clf.predict(X)
                print(f"\t \t \t % active: {np.mean(y_class)}")
                X = X[y_class == 1]
            
            #print(f'Predicting {len(X)} samples')
            if len(X) != 0:
                y_pred = model.predict(X)
                date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(results_file, 'a') as f:
                    f.write(f'{date},{model.__class__.__name__},{clf.__class__.__name__},{dataset},{design},{eye},{experiment},{y_pred.tolist()}\n')
                box = ax.boxplot([y_pred], showfliers=False, vert=False, patch_artist=True, positions=[j*3 + (k+1)], widths=0.6, label=experiment)
                for item in ['boxes', 'whiskers', 'caps']:
                    plt.setp(box[item], color=colors[experiment])
                plt.setp(box['means'], color='black')
                plt.setp(box['medians'], color='black')
            else:
                with open(results_file, 'a') as f:
                    f.write(f'{date},{model.__class__.__name__},{clf.__class__.__name__},{dataset},{design},{eye},{experiment},\n')
                # add empty box
                box = ax.boxplot([[]], showfliers=False, vert=False, patch_artist=True, positions=[j*3 + (k+1)], widths=0.6, label=experiment)
                for item in ['boxes', 'whiskers', 'caps']:
                    plt.setp(box[item], color=colors[experiment])
                plt.setp(box['means'], color='black')
                plt.setp(box['medians'], color='black')

            boxes.append(box)

        ax.set_yticks(np.arange(2, len(datasets)*3, 3))
        if eye_idx % 3 == 0:
            ax.set_yticklabels([get_label_from_ds(x) for x in datasets])
        else:
            ax.set_yticklabels([])
        
        
    # horizontal lines between yticklabels
    for l in range(1, 12):
        ax.axhline(y=l*3+0.5, color='grey', linewidth=0.5, linestyle='--')
    if eye == 6:
        ax.set_xlabel('Predicted AP counts')

    if eye == 1 or eye == 4:
        ax.text(22, 37, 'White\n SNR', fontsize=7)
        ax.text(32, 37, ' ME\nSNR', fontsize=7)

fig.legend([x['whiskers'][0] for x in boxes[-3:]], ['With Stimulus', 'Without Stimulus', 'TTX'], ncols=3,
            loc='upper center', bbox_to_anchor=(0.39, 0))

save_figure(f"predict_real_data_{design}", width=6, height=9) # take up a whole page