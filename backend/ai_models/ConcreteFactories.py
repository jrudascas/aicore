from .chestxnet.ChestXNetModel import ChestXNetModel, OtherModel
from .AbstractFactories import AbstractModelFactory
from .chestxnet.ChestXNetConstanteManager import PATH_WEIGHTS


class ChestXNetModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = PATH_WEIGHTS
        return ChestXNetModel(weights)


class OtherModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = None
        return OtherModel(weights)
