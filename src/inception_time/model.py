import torch
import numpy as np

from src.inception_time.modules import InceptionModel


class InceptionTime():
    
    def __init__(self,
                 filters=32,
                 depth=6,
                 n_models=5,
                 learning_rate= 0.01,
                 batch_size=32,
                 epochs = 3,
                 verbose=True):
        
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
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.
        
        models: int.
            The number of models.
        '''
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.filters = filters
        self.depth = depth
        self.n_models = n_models

    
    def fit(self,
            x, 
            y):
        
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
        # Scale the data.
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        self.sigma = np.nanstd(x, axis=0, keepdims=True)
        x = (x - self.mu) / self.sigma
        
        # Save the data.
        self.x = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)
        
        # Build and save the models.
        self.models = [
            InceptionModel(
                input_size=x.shape[1],
                filters=self.filters,
                depth=self.depth,
            ).to(self.device) for _ in range(self.n_models)
        ]

        # Generate the training dataset.
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for m in range(len(self.models)):
            
            # Define the optimizer.
            optimizer = torch.optim.Adam(self.models[m].parameters(), lr=self.learning_rate)
            
            # Define the loss function.
            loss_fn = torch.nn.MSELoss()
            
            # Train the model
            print(f'Training model {m + 1} on {self.device}.')
            self.models[m].train(True)
            for epoch in range(self.epochs):
                for features, target in dataset:
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    loss = loss_fn(output, target.float().to(self.device))
                    loss.backward()
                    optimizer.step()
                    #accuracy = (torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1) == target).float().sum() / target.shape[0]
                    rmse = torch.sqrt(((output - target)**2).sum())
                if self.verbose:
                    print('epoch: {}, loss: {:,.6f}, rmse: {:.6f}'.format(1 + epoch, loss, rmse))
            self.models[m].train(False)
            print('-----------------------------------------')
    
    def predict(self, x):
        # TODO: is_fitted()
        
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
        y: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        '''

        # Scale the data.
        x = torch.from_numpy((x - self.mu) / self.sigma).float().to(self.device)
        
        # Get the predicted probabilities.
        with torch.no_grad():
            output = torch.concat([torch.nn.functional.softmax(model(x), dim=-1).unsqueeze(-1) for model in self.models], dim=-1).mean(-1)
        
        # Get the predicted labels.
        y = output.detach().cpu().numpy().flatten()

        return y



if __name__ == '__main__':
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error

    dataset = '../../simulated_data/DS_80_10_100'
    X = np.load(os.path.join(dataset, "X.npy"))[:, np.newaxis, :]
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, y_train.shape)

    model = InceptionTime()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(root_mean_squared_error(y_pred, y_test))