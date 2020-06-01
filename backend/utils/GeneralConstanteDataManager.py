from ..utils.utils import find_model
from ..exceptions.exceptions_generator import ModelNotFoundException
from os import path
from ..models import *

REQUEST_STATUS_RECEIVED = 'RECEIVED'
REQUEST_STATUS_PROCESSING = 'PROCESSING'
REQUEST_STATUS_FINISHED_SUCCESSFULLY = 'FINISHED_SUCCESSFULLY'
REQUEST_STATUS_FINISHED_WITH_ERRORS = 'FINISHED_WITH_ERRORS'

REQUEST_STATUS_LIST = [REQUEST_STATUS_RECEIVED, REQUEST_STATUS_PROCESSING, REQUEST_STATUS_FINISHED_SUCCESSFULLY,
                       REQUEST_STATUS_FINISHED_WITH_ERRORS]

REQUEST_TYPE_NAME_PREDICTION = 'PREDICTION'

REQUEST_TYPE_NAME_LIST = [REQUEST_TYPE_NAME_PREDICTION]

if path.exists('db.sqlite3'):

    try:
        model = Model.objects.filter(name='Fred')

        try:
            find_model(Model, 'name', 'ChestXNetv1.0')
        except ModelNotFoundException:
            data = {'name': 'ChestXNetv1.0',
                    'model_factory_name': 'ChestXNetModelFactory'}
            Model.objects.create(**data)

        try:
            find_model(Model, 'name', 'COVID19v1.0')
        except ModelNotFoundException:
            data = {'name': 'COVID19v1.0',
                    'model_factory_name': 'Covid19ModelFactory'}
            Model.objects.create(**data)

        try:
            find_model(Model, 'name', 'COVID19CTv1.0')
        except ModelNotFoundException:
            data = {'name': 'COVID19CTv1.0',
                    'model_factory_name': 'Covid19CTModelFactory'}
            Model.objects.create(**data)

        for REQUEST_STATUS in REQUEST_STATUS_LIST:
            try:
                find_model(ModelRequestStatus, 'name', REQUEST_STATUS)
            except ModelNotFoundException:
                data = {'name': REQUEST_STATUS}
                ModelRequestStatus.objects.create(**data)

        for REQUEST_TYPE_NAME in REQUEST_TYPE_NAME_LIST:

            try:
                find_model(ModelRequestType, 'name', REQUEST_TYPE_NAME)
            except ModelNotFoundException:
                data = {'name': REQUEST_TYPE_NAME}
                ModelRequestType.objects.create(**data)

    except Exception as e:
        pass