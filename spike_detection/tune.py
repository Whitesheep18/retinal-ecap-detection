
if __name__ == "__main__":
    import argparse
    import os
    import sys
    from src.train_and_eval import train_and_eval
    from src.utils import sorting_key
    from src.inception_time.model import InceptionTime
    from numpy.random import choice

    parser = argparse.ArgumentParser(description='Train a model on the simulated data')
    parser.add_argument('--results', type=str, default='results.csv', help='results_file [.csv]')
    parser.add_argument('--save_model_path', type=str, default='False', help='wether to save the model as pickle [<path_to_model>/False]')
    parser.add_argument('--dataset', type=str, help='path to dataset folder')
    parser.add_argument('--dataset_idx', type=int, help='index of the dataset in the folder')
    parser.add_argument('--verbose', type=int, default=1, help="print what's going on")
    parser.add_argument('--comment', type=str, default='', help="anything else (eg. jobid) you want to add")
    parser.add_argument('--hp_tune_type', type=str, default='random', choices=['random', 'grid'], help="type of tuning. either 'grid' or 'random' search")
    parser.add_argument('--num_random_hp_comb', type=int, default=5, help="number of HP combinations if hp_tune_type is 'random'")
    # inception time arguments
    parser.add_argument('--n_models', type=int, default=1, help='number of models in inception time')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs in InceptionTime')
    parser.add_argument('--learning_rate_list', type=float, nargs='*', default=[0.0001, 0.001],help='learning rate in InceptionTime')
    parser.add_argument('--dropout_list', type=float, nargs='*', default=[0.2, 0.5, 0.8], help='portion of weights to forget in InceptionTime')
    parser.add_argument('--l2_penalty_list', type=float, nargs='*', default=[0.0001, 0.001], help='l2 penalty in InceptionTime')
    parser.add_argument('--init_stride_list', type=int, nargs='*', default=[-1, 2], help='stride of initial cnn. If zero or less, no initial cnn is applied')
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
            model = InceptionTime(verbose=args.verbose, epochs=args.n_epochs,n_models=args.n_models, **hp_comb)
            train_and_eval(model, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)
            hp_combs.append(hp_comb)

    elif args.hp_tune_type == 'grid':
        for learning_rate in args.learning_rate_list:
            for dropout in args.dropout_list:
                for l2_penalty in args.l2_penalty_list:
                    for init_stride in args.init_stride_list:
                        model = InceptionTime(verbose=args.verbose, n_models=args.n_models, epochs=args.n_epochs, 
                                              learning_rate=learning_rate, dropout=dropout, l2_penalty=l2_penalty, init_stride=init_stride)
                        train_and_eval(model, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)
    else:
        print("Unknown hp_tune_type. Choose")



