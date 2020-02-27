import django_filters
from .models import *


class ModelResponseFilter(django_filters.FilterSet):
    test = django_filters.CharFilter(field_name='response', lookup_expr='isnull')

    class Meta:
        model = ModelResponse
        fields = ['uuid', 'test']