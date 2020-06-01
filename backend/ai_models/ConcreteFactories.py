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


class Covid19CTModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        from .covid19ct.Covid19CTConstanteManager import COVID19_CT_PATH_WEIGHTS
        from .covid19ct.Covid19CTModel import Covid19CTModel
        weights = COVID19_CT_PATH_WEIGHTS
        return Covid19CTModel(weights)
