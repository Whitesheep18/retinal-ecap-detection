class AveragePredictionModel:
    def fit(self, x_train, y_train):
        "no training required. this is an unsupervised method."
        pass

    def predict(self,y_train, y_test):
        avg_value = sum(y_train)/len(y_train)
        return [avg_value]*len(y_test)
    
    def get_params(self):
        return {}
