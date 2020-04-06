from .chestxnet.ChestXNetModel import ChestXNetModel
from .covid19.Covid19Model import Covid19Model
from .AbstractFactories import AbstractModelFactory
from .chestxnet.ChestXNetConstanteManager import CHESTXNET_PATH_WEIGHTS
from .covid19.Covid19ConstanteManager import COVID19_PATH_WEIGHTS, COVID19_PATH_MODEL_METADATA


class ChestXNetModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = CHESTXNET_PATH_WEIGHTS
        return ChestXNetModel(weights)


class Covid19ModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = COVID19_PATH_WEIGHTS
        metadata = COVID19_PATH_MODEL_METADATA
        return Covid19Model(weights, metadata)
