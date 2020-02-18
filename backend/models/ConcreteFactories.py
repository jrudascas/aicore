from .chestxnet.ChestXNetModel import ChestXNetModel, OtherModel
from .AbstractFactories import AbstractModelFactory


class ChestXNetModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self, weights):
        return ChestXNetModel(weights)


class OtherModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self, weights):
        return OtherModel(weights)
