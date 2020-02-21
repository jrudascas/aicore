from .chestxnet.ChestXNetModel import ChestXNetModel, OtherModel
from .AbstractFactories import AbstractModelFactory


class ChestXNetModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = '/home/jrudascas/PycharmProjects/aicore/backend/ai_models/chestxnet/pretrained/checkpoint'
        return ChestXNetModel(weights)


class OtherModelFactory(AbstractModelFactory):

    def __init__(self):
        pass

    def create_model(self):
        weights = None
        return OtherModel(weights)
