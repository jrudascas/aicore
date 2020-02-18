from .chestxnet.ChestXNetModel import ChestXNetModel, OtherModel


class AbstractModelFactory:
    def __init__(self):
        pass

    def create_model(self, weights):
        pass