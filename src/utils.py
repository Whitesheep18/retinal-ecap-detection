import numpy as np
import neo
import os
import re

def get_signal(filepath='../data/10Hz_1V_neg500mV_1ms003.ns5', verbose=1):
    reader = neo.io.BlackrockIO(filename=filepath)
    if verbose: print("No. blocks", reader.block_count())
    block = reader.read_block()
    if verbose: print("Block size", block.size)
    segment = block.segments[0]
    if verbose: print("Segment size", segment.size)
    signal = segment.analogsignals[0]
    if verbose: print("Signal shape (timestampts, signals)", signal.shape)
    time = np.array(signal.times)
    if verbose: print(f"Time duration {time[-1]}s")
    return time, signal.magnitude


def get_signal_by_type(path_to_data='../data', eye=1, design='3D', experiment='TTX', verbose=1):
    path = f'{path_to_data}/{design}/Eye {eye}'
    files = os.listdir(path)
    if experiment == 'TTX':
        file = [f for f in files if 'TTX' in f][0]
    elif experiment == 'non-stimulated':
        file = [f for f in files if 'NO_STIM' in f][0]
    elif experiment == 'stimulated':
        file = [f for f in files if 'TTX' not in f and 'NO_STIM' not in f][0]
    else:
        raise Exception(f"Experiment {experiment} not found. Use either 'TTX', 'non-stimulated' or 'stimulated'")
    if verbose: 
        print(f"Reading {file}")
    return get_signal(filepath=f'{path}/{file}', verbose=verbose)

def get_template(template_type, path_to_templates = '../retinal-ecap-detection/simulate'):
    if template_type == 'SA':
        return np.load(f"{path_to_templates}/SA_templates.npy")
    elif template_type == 'ME':
        return np.load(f"{path_to_templates}/ME_template.npy")
    elif template_type == 'AP':
        return np.load(f"{path_to_templates}/AP_templates.npy")
    else:
        raise Exception(f"Template {template_type} not found. Use either 'SA', 'ME' or 'AP'")

# Define a sorting key based on SNR and ME values
def sorting_key(dataset_name):
    match = re.match(r"DS_(-?\d+)_(-?\d+)_10", dataset_name)
    if match:
        snr = int(match.group(1))
        me = int(match.group(2))
        return (snr, me)  
    return (float('inf'), float('inf'))
