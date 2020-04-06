class BaseModel(object):
    def __init__(self, weights, model_metadata = None):
        self.weights = weights
        self.model_metadata = model_metadata

    def run_prediction(self, metadata):
        raise NotImplementedError()

    def run_training(self):
        raise NotImplementedError()

    def run_evaluation(self):
        raise NotImplementedError()


