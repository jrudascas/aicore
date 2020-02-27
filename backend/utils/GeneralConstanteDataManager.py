from ..models import ModelRequestType, ModelRequestStatus
from ..utils.utils import find_model
from ..exceptions.exceptions_generator import ModelNotFoundException


REQUEST_STATUS_RECEIVED = 'RECEIVED'
REQUEST_STATUS_PROCESSING = 'PROCESSING'
REQUEST_STATUS_FINISHED_SUCCESSFULLY = 'FINISHED_SUCCESSFULLY'
REQUEST_STATUS_FINISHED_WITH_ERRORS = 'FINISHED_WITH_ERRORS'

REQUEST_STATUS_LIST = [REQUEST_STATUS_RECEIVED, REQUEST_STATUS_PROCESSING, REQUEST_STATUS_FINISHED_SUCCESSFULLY, REQUEST_STATUS_FINISHED_WITH_ERRORS]

REQUEST_TYPE_NAME_PREDICTION = 'PREDICTION'

REQUEST_TYPE_NAME_LIST = [REQUEST_TYPE_NAME_PREDICTION]

#id database exist
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