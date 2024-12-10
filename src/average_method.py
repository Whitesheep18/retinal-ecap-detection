class AveragePrediction:
    def fit(self, x_train, y_train):
        "no training required. this is an unsupervised method."
        pass

    def predict(self,y_train, x_test):
        avg_value = sum(y_train)/len(y_train)
        return [avg_value]*len(x_test)
    
    def get_params(self):
        return {}
    
    def __repr__(self):
        return f"AveragePrediction()"


if __name__=='__main__':
    model = AveragePrediction()
    print(model)

