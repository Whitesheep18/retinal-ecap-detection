class Filter:
    def fit(self, x_train, y_train):
        "no training"
        pass

    def predict(self,y_train, x_test):
        pass
    
    def get_params(self):
        return {}
    
    def __repr__(self):
        return f"Filter()"


if __name__=='__main__':
    model = Filter()
    print(model)

