from rest_framework import viewsets
from rest_framework.response import Response
from .utils.utils import create_response, find_model, update_model_status
from rest_framework import status
from django.utils.decorators import method_decorator
from drf_yasg.utils import swagger_auto_schema
from .serializers import *
import importlib
from .utils.GeneralConstanteDataManager import REQUEST_STATUS_RECEIVED, REQUEST_STATUS_PROCESSING, REQUEST_STATUS_FINISHED_SUCCESSFULLY, REQUEST_STATUS_FINISHED_WITH_ERRORS, REQUEST_TYPE_NAME_PREDICTION


module = importlib.import_module("backend.ai_models.ConcreteFactories")


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer


class ModelViewSet(viewsets.ModelViewSet):
    queryset = Model.objects.all()
    serializer_class = ModelSerializer
    filter_fields = ('uuid', 'name')


class ModelResponseViewSet(viewsets.ModelViewSet):
    queryset = ModelResponse.objects.all()
    serializer_class = ModelResponseSerializer
    filter_fields = ('request__images__private_id', )


@method_decorator(name='list', decorator=swagger_auto_schema(
    operation_description="description from swagger_auto_schema via method_decorator",
))
class ModelRequestViewSet(viewsets.ModelViewSet):
    queryset = ModelRequest.objects.all()
    serializer_class = ModelRequestSerializer

    @swagger_auto_schema(operation_description="GET /request/{id}/")
    def retrieve(self, request, *args, **kwargs):
        super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(operation_description="GET /request/ - This operation return a list of all Request objects")
    def list(self, request, *args, **kwargs):
        super().list(request, *args, **kwargs)

    @swagger_auto_schema(operation_description="PUT /request/ - This operator allows to create a new ")
    def create(self, request, *args, **kwargs):

        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            images_data = serializer.data['images']

            model = find_model(Model, 'uuid', serializer.data['model'])
            model_request_dict = {'user': request.user,
                                  'model': model,
                                  'status': find_model(ModelRequestStatus, 'name', REQUEST_STATUS_RECEIVED),
                                  'type': find_model(ModelRequestType, 'name', REQUEST_TYPE_NAME_PREDICTION)}

            model_request = ModelRequest.objects.create(**model_request_dict)
            for image_data in images_data:
                Image.objects.create(request=model_request, **image_data)

        except Exception as e:
            if 'model_request' in locals():
                update_model_status(model_request, REQUEST_STATUS_FINISHED_WITH_ERRORS)

            return Response(data=create_response(data=None, status='BAD',
                                                 comments='Impossible create the related models. Details: ' + e.__str__()),
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            model_factory_name = model.model_factory_name
            model_factory_name = getattr(module, model_factory_name)

            abstract_factory = model_factory_name()

            ai_model = abstract_factory.create_model()

            model_request = update_model_status(model_request, REQUEST_STATUS_PROCESSING)

            prediction_response = ai_model.run_prediction(images_data)

            model_response = {'request': model_request, 'response':prediction_response}
            ModelResponse.objects.create(**model_response)
            model_request = update_model_status(model_request, REQUEST_STATUS_FINISHED_SUCCESSFULLY)

            return Response(data=create_response(
                data=prediction_response, status='OK'), status=status.HTTP_200_OK)

        except Exception as e:

            model_request = update_model_status(model_request, REQUEST_STATUS_FINISHED_WITH_ERRORS)

            return Response(data=create_response(data=None, status='BAD', comments=e.__str__()),
                            status=status.HTTP_400_BAD_REQUEST)
