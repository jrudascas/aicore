from ..exceptions.exceptions_generator import ModelNotFoundException
from ..models import ModelRequestStatus


def create_response(data, status, comments=''):
    return {'data': data, 'status': status, 'comments': comments}


def update_model_status(model, status):
    if model:
        model.status = find_model(ModelRequestStatus, 'name', status)
        model.save()

    return model


def find_model(model_type, field_name, value_name):
    query_filter = {field_name: value_name}
    # query_filter = Q({field_name: value_name})
    query_dict = model_type.objects.filter(**query_filter)
    if not query_dict:
        raise ModelNotFoundException(
            '{0} not found by using the filter {1} = {2}'.format(model_type, field_name, value_name))

    return query_dict[0]