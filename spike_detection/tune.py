
def check_experiment_exists(args, dataset_path, params, tune_results):
    exists = False
    for _, row in tune_results.iterrows():
        p = row['params'] == params
        d = row['Dataset'] == dataset_path.split('/')[-1]
        c = row['Classification model'] == args.classification_model
        if  p and d and c:
           exists = True

    return exists


if __name__ == "__main__":
    import argparse
    import os
    import sys
    from src.train_and_eval import train_and_eval
    from src.utils import sorting_key, switch_quotes
    from src.inception_time.model import InceptionTimeE
    from numpy.random import choice
    import pandas as pd
    import json

    parser = argparse.ArgumentParser(description='Train a model on the simulated data')
    parser.add_argument('--classification_model', type=str, required=True, choices = ['Filter', 'RandomForestClassifier', 'HIVECOTEV2'], default='Filter', help='classification model or models to train')
    parser.add_argument('--results', type=str, default='results_tuning.csv', help='results_file [.csv]')
    parser.add_argument('--save_model_path', type=str, default='False', help='wether to save the model as pickle [<path_to_model>/False]')
    parser.add_argument('--dataset', type=str, help='path to dataset folder')
    parser.add_argument('--dataset_idx', type=int, help='index of the dataset in the folder')
    parser.add_argument('--verbose', type=int, default=1, help="print what's going on")
    parser.add_argument('--comment', type=str, default='', help="anything else (eg. jobid) you want to add")
    parser.add_argument('--hp_tune_type', type=str, default='random', choices=['random', 'grid'], help="type of tuning. either 'grid' or 'random' search")
    parser.add_argument('--num_random_hp_comb', type=int, default=5, help="number of HP combinations if hp_tune_type is 'random'")
    # inception time arguments
    parser.add_argument('--n_models', type=int, default=1, help='number of models in InceptionTimeE')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs in InceptionTime')
    parser.add_argument('--depth', type=int, default=6, help='number of layers in inceptionTime')
    parser.add_argument('--min_n_epochs_list', type=int, nargs='*', default=[1], help='minimum number of epochs before early stopping in InceptionTime')
    parser.add_argument('--learning_rate_list', type=float, nargs='*', default=[0.0001, 0.001],help='learning rate in InceptionTime')
    parser.add_argument('--dropout_list', type=float, nargs='*', default=[0.2, 0.5, 0.8], help='portion of weights to forget in InceptionTime')
    parser.add_argument('--l2_penalty_list', type=float, nargs='*', default=[0.0001, 0.001], help='l2 penalty in InceptionTime')
    parser.add_argument('--init_stride_list', type=int, nargs='*', default=[-1, 2], help='stride of initial cnn in InceptionTime. If zero or less, no initial cnn is applied')
    args = parser.parse_args()

    if args.dataset_idx is not None:
        datasets = [x for x in os.listdir('simulated_data') if x.startswith('DS')]
        datasets = sorted(datasets, key=sorting_key)
        dataset_path = os.path.join('simulated_data', datasets[args.dataset_idx-1])
    elif args.dataset is not None:
        dataset_path = os.path.join('simulated_data', args.dataset)
    else:
        print("Either dataset or dataset_idx must be given")
        sys.exit(1)

    if args.classification_model == "Filter":
        from src.filter_classification import Filter
        classifier = Filter()
    elif args.classification_model == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
    elif args.classification_model == "HIVECOTEV2":
        from aeon.classification.hybrid import HIVECOTEV2
        args.classification_model = HIVECOTEV2(verbose=1, time_limit_in_minutes = 1)
    else:
        print(f"Unknown classifcation model {args.classification_model}")
        sys.exit(1)

    if os.path.exists(args.results):
        tune_results = pd.read_csv(args.results)
        tune_results['params'] = tune_results['params'].apply(lambda x: json.loads(switch_quotes(x)))
    else:
        tune_results = None

    if args.hp_tune_type == 'random':
        if args.num_random_hp_comb > len(args.learning_rate_list)*len(args.dropout_list)*len(args.l2_penalty_list):
            print('Too many combinations')
            sys.exit(1)

        hp_combs = []
        while len(hp_combs) < args.num_random_hp_comb:

            # choose comb
            hp_comb =  {"learning_rate": choice(args.learning_rate_list), 
                        "dropout": choice(args.dropout_list),
                        "l2_penalty": choice(args.l2_penalty_list),
                        "init_stride": choice(args.init_stride_list)}
            if hp_comb in hp_combs:
                continue

            # train model
            print('hyperparameters', hp_comb)
            model = InceptionTimeE(verbose=args.verbose, epochs=args.n_epochs,n_models=args.n_models, 
                                  depth=args.depth,filters=32,batch_size=64,optimizer='AdamW',**hp_comb)

            params = model.get_params()
            print("Params", params)

            if tune_results is not None and check_experiment_exists(args, dataset_path, params, tune_results):
                print('old model! continuing')
                continue
        
            print('new model!')

            train_and_eval(model, classifier, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)
            hp_combs.append(hp_comb)

    elif args.hp_tune_type == 'grid':
        for min_n_epochs in args.min_n_epochs_list:
            for learning_rate in args.learning_rate_list:
                for dropout in args.dropout_list:
                    for l2_penalty in args.l2_penalty_list:
                        for init_stride in args.init_stride_list:

                            model = InceptionTimeE(verbose=args.verbose, n_models=args.n_models, epochs=args.n_epochs, min_epochs = min_n_epochs,
                                                learning_rate=learning_rate, dropout=dropout, l2_penalty=l2_penalty, init_stride=init_stride,
                                                depth=args.depth, filters=32, batch_size=64, optimizer='AdamW')
                            
                            params = model.get_params()
                            print("Params", params)
                            
                            if tune_results is not None and check_experiment_exists(args, dataset_path, params, tune_results):
                                print('old model! continuing')
                                continue
                        
                            print('new model!')

                            train_and_eval(model, classifier, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)
    else:
        print("Unknown hp_tune_type. Choose")
