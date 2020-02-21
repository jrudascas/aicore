from rest_framework import viewsets
from django.contrib.auth.models import User
from backend.serializers import UserSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from backend.models import *
from .utils.utils import create_response
from rest_framework import status
from .ai_models.ConcreteFactories import ChestXNetModelFactory
from rest_framework.authentication import TokenAuthentication
from django.contrib.auth.models import User
import json


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer


@api_view(['PATCH'])
def get_request(request):
    if request.method == 'PATCH':
        try:
            body = request.data.copy()

            # check_json_structure()
            # if not check_json_structure():
            #    raise Exception

            model = Model.objects.filter(name=body['model']['name'])
            if not model:
                raise Exception('Model {0} in the version {1} does not exist'.format(body['model']['name']))

            data = {'user': request.user, 'model': model[0], 'type': ModelRequestType.objects.filter(id=1)[0],
                    'status': ModelRequestStatus.objects.filter(id=1)[0]}
            model_request = ModelRequest.objects.create(**data)

            metadata = body['metadata']

            model_factory_name = model[0].model_factory_name

            import importlib
            module = importlib.import_module("backend.ai_models.ConcreteFactories")
            model_factory_name = getattr(module, model_factory_name)

            abstract_factory = model_factory_name()

            ai_model = abstract_factory.create_model()
            prediction_response = ai_model.run_prediction(metadata)

            data = {'request': model_request, 'response': json.dumps(prediction_response)}
            ModelResponse.objects.create(**data)

            return Response(data=create_response(data=prediction_response, status='OK'), status=status.HTTP_200_OK)

        except Exception as e:
            return Response(data=create_response(data=None, status='BAD', comments=e.__str__()),
                            status=status.HTTP_400_BAD_REQUEST)
