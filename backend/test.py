from ai_models.ConcreteFactories import ChestXNetModelFactory
from ai_models.chestxnet.ChestXNetConstanteManager import PATH_WEIGHTS


def test_factory(abstract_factory):
    model = abstract_factory.create_model(weights=PATH_WEIGHTS)
    print(model.run_prediction('/home/jrudascas/PycharmProjects/datasets/images_01/images/00000013_003.png'))


factory = ChestXNetModelFactory()
test_factory(factory)