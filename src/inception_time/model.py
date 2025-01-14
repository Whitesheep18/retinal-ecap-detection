import torch
import numpy as np
import pickle
from src.inception_time.modules import InceptionModel
from torch.optim import Adam, AdamW
from tqdm import tqdm

class InceptionTimeE():
    
    def __init__(self,
                 filters=32,
                 depth=6,
                 init_stride=-1,
                 n_models=5,
                 learning_rate= 0.0001,
                 batch_size=32,
                 epochs = 3,
                 min_epochs = 1,
                 dropout = 0,
                 l2_penalty = 0,
                 optimizer = 'Adam',
                 verbose=True
                 ):
        
        '''
        Adapted to regression from flaviagianmario
        Implementation of InceptionTime model introduced in Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier,
        C., Schmidt, D.F., Weber, J., Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. InceptionTime:
        Finding AlexNet for Time Series Classification. Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

        Parameters:
        __________________________________

        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Maximum number of epochs.

        min_epochs: int.
            Minimum number of epochs before early stopping. If min_epochs > epochs, no early stopping is applied

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.
        
        init_stride: int
            The stride of an initial CNN layer. The purpose is to have an initial downsampling. 
            If None, no downsampling is applied before the inception blocks
        
        models: int.
            The number of models.
        '''
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.verbose = verbose
        self.filters = filters
        self.init_stride = init_stride
        self.depth = depth
        self.n_models = n_models
        self.optimizer = optimizer
        self.train_loss = [[] for i in range(n_models)]
        self.valid_loss = [[] for i in range(n_models)]

        self.is_fitted = False


    def _set_scalers(self, x):
        self.mu = np.nanmean(x)
        self.sigma = np.nanstd(x)

    def _build_models(self):
        self.models = [
            InceptionModel(
                input_size=1, # channels
                filters=self.filters,
                depth=self.depth,
                dropout=self.dropout,
                init_stride=self.init_stride
            ).to(self.device) for _ in range(self.n_models)
        ]

    
    def fit(self,
            x, 
            y,
            x_val,
            y_val,
            train_from_scratch=True):
        
        '''
        Train the models.

        Parameters:
        __________________________________

        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        y: np.array.
            True response values shape (samples,) where samples is the number of time series.

        '''
        # Add channels dimention
        x = x[:, np.newaxis, :]
        x_val = x_val[:, np.newaxis, :]


        if train_from_scratch:
            self._set_scalers(x) 
            self._build_models()

        # Scale the data
        x = (x - self.mu) / self.sigma
        x_val = (x_val - self.mu) / self.sigma

        # Save the data.
        self.x = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)
        self.x_val = torch.from_numpy(x_val).float().to(self.device)
        self.y_val = torch.from_numpy(y_val).long().to(self.device)


        # Generate the training dataset.
        train_dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=False
        )


    
        for m in range(len(self.models)):
            
            # Define the optimizer.
            if self.optimizer == 'Adam':
                optimizer = Adam(self.models[m].parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
            elif self.optimizer == 'AdamW':
                optimizer = AdamW(self.models[m].parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)

            # Define the loss function.
            loss_fn = torch.nn.MSELoss()
            
            # Train the model
            print(f'Training model {m + 1} on {self.device}.')
            self.models[m].train(True)

            num_steps = len(train_dataset)*self.epochs
            epoch = 0
            # Early stopping parameters
            patience = 10
            moving_avg_window = 10
            best_val_loss = float('inf')
            patience_counter = 0
            val_loss_history = [float('inf')] * moving_avg_window
            moving_avg_val_loss = float('inf')

            with tqdm(range(num_steps)) as pbar:
                running_loss = 0.0
                epoch_loss = 0.0
                avg_val_loss = 0.0
                for step in pbar:
                    features, target = next(iter(train_dataset))
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    output = output.flatten()
                    loss = loss_fn(output, target.float().to(self.device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # Report
                    if step % 10 ==0 and self.verbose:
                        loss = loss.detach().cpu()
                        pbar.set_description(f"epoch={epoch+1}, step={step}, current_loss={loss:.1f}, epoch_loss={epoch_loss:.1f}, moving_valid_loss={moving_avg_val_loss:.1f}")

                    if (step+1) % len(train_dataset) == 0:
                        epoch_loss = running_loss/len(train_dataset)
                        running_loss = 0.0
                        self.train_loss[m].append(epoch_loss)

                        # Validation
                        self.models[m].eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for X_val, y_val in val_dataset:
                                val_output = self.models[m](X_val.to(self.device)).flatten()
                                val_loss += loss_fn(val_output, y_val.float().to(self.device)).item()
                        self.models[m].train(True)
                        avg_val_loss = val_loss / len(val_dataset)
                        self.valid_loss[m].append(avg_val_loss)
                        val_loss_history.append(avg_val_loss)
                        val_loss_history.pop(0)  # Maintain a fixed-size window
                        moving_avg_val_loss = np.mean(val_loss_history)
                        epoch += 1

                        if epoch > self.min_epochs:
                            # Early stopping logic
                            if moving_avg_val_loss < best_val_loss:
                                best_val_loss = moving_avg_val_loss
                                patience_counter = 0 
                            else:
                                patience_counter += 1
                                if patience_counter >= patience:
                                    print(f"Early stopping at epoch {epoch}, no improvement in validation loss for {patience} epochs.")
                                    break
                            

            self.models[m].train(False)
            self.is_fitted = True

    def is_fitted(self):
        return self.fitted
    
    def predict(self, x):

        '''
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        Returns:
        __________________________________
        y_avg: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
            
        y_individual: np.array
            Predicted labels for each model in self.models
        '''

        x = x[:, np.newaxis, :]
        x = torch.from_numpy((x - self.mu) / self.sigma).float()

        test_dataset = torch.utils.data.DataLoader(
                        dataset=torch.utils.data.TensorDataset(x),
                        batch_size=self.batch_size,
                        shuffle=False
                        )
        y_avg, y_individual = [], []

        num_steps = len(test_dataset)
        with tqdm(range(num_steps)) as pbar:
            tds = iter(test_dataset)
            for step in pbar:
                features = next(tds)[0]

                # Get the predicted probabilities.
                with torch.no_grad():
                    outputs = torch.concat([model(features.to(self.device)) for model in self.models], dim=-1)

                # Convert outputs to numpy array for individual predictions.
                y_individual_batch = outputs.detach().cpu().numpy()
                y_individual.append(y_individual_batch)

                # Calculate the average prediction.
                y_avg_batch = y_individual_batch.mean(axis=-1).flatten()
                y_avg.append(y_avg_batch)

                if step % 10 ==0 and self.verbose:
                    pbar.set_description(f"batch={step}")

        y_avg = np.concatenate(y_avg)
        y_individual = np.concatenate(y_individual)
        return y_avg, y_individual
    
    def save(self, file_path):

        meta_file_path = f"{file_path}_metadata.pkl"
        params = self.get_params()

        with open(meta_file_path, "wb") as f:
            pickle.dump(params, f)

        mu_file_path = f"{file_path}_mu.npy"
        with open(mu_file_path, "wb") as f:
            pickle.dump(self.mu, f)

        sigma_file_path = f"{file_path}_sigma.npy"
        with open(sigma_file_path, "wb") as f:
            pickle.dump(self.sigma, f)

        for n in range(self.n_models):
            model_file_path = f"{file_path}_model{n}.pkl"
            model = self.models[n]
            torch.save(model.state_dict(), model_file_path)

    def load(self, file_path):
        print('loaded model is inteded for prediction only! ')
        meta_file_path = f"{file_path}_metadata.pkl"
        with open(meta_file_path, "rb") as f:
            params = pickle.load(f)
        self.set_params(params)

        mu_file_path = f"{file_path}_mu.npy"
        with open(mu_file_path, "rb") as f:
            self.mu = pickle.load(f)

        sigma_file_path = f"{file_path}_sigma.npy"
        with open(sigma_file_path, "rb") as f:
            self.sigma = pickle.load(f)

        print('loading models')

        self.models = []

        for n in range(self.n_models):
            model_file_path = f"{file_path}_model{n}.pkl"
            model = InceptionModel(input_size=1,
                                   filters=params['filters'],
                                   depth=params['depth'])
            model.load_state_dict(torch.load(model_file_path, weights_only=True))
            model.eval()
            self.models.append(model.to(params['device']))
            
    def get_params(self):
        return {'device': self.device.type,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'min_epochs': self.min_epochs,
                'dropout': self.dropout,
                'l2_penalty': self.l2_penalty,
                'verbose': self.verbose,
                'filters':  self.filters,
                'init_stride': self.init_stride,
                'depth': self.depth,
                'n_models': self.n_models,
                'optimizer': self.optimizer
                }

    def set_params(self, params):
        self.device = torch.device('cuda:0' if params['device']=='cuda' else 'cpu')
        self.learning_rate = params["learning_rate"]
        self.dropout = params["dropout"]
        self.l2_penalty = params["l2_penalty"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.min_epochs = params["min_epochs"]
        self.verbose = params["verbose"]
        self.filters = params["filters"]
        self.init_stride = params["init_stride"]
        self.depth = params["depth"]
        self.n_models = params["n_models"]
        self.optimizer = params["optimizer"]


if __name__ == '__main__':
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = 'simulated_data/DS_80_80_10'
    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    model = InceptionTimeE(n_models = 3, epochs=150, init_stride=2, depth=6, dropout=0.0, learning_rate=0.0001)
    model.fit(X_train, y_train, X_val, y_val)
    print('fit ok')
    y_pred, y_individual = model.predict(X_test)
    print(root_mean_squared_error(y_test, y_pred))
    

    from src.visualize.training import plot_loss
    plot_loss(model.train_loss, model.valid_loss)
    # print(y_pred)
    from src.visualize.results import residual_plot_individual
    residual_plot_individual(y_test=y_test, y_individual=y_individual)

    # model.save('../../models/InceptionTime_DS_80_10_10')

    # model2 = InceptionTimeE()
    # model2.load('../../models/InceptionTime_DS_80_10_10')
    
