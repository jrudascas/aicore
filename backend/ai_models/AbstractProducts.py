class BaseModel(object):
    def __init__(self, weights):
        self.weights = weights

    def run_prediction(self, metadata):
        raise NotImplementedError()

    def run_training(self):
        raise NotImplementedError()

    def run_evaluation(self):
        raise NotImplementedError()


