from rest_framework import serializers
from .models import *


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'is_staff']


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['private_id', 'url', 'array']
        extra_kwargs = {
            'private_id': {
                'help_text': 'Private ID from the client system'},
            'url': {
                'help_text': "Public URL in which the image is accessible"
            },
            'array': {
                'help_text': "A numpy array as list that represents a gray-scale image"
            },
            'read_only_nullable': {'allow_null': True},
        }


class ModelRequestSerializer(serializers.ModelSerializer):
    images = ImageSerializer(many=True)

    class Meta:
        model = ModelRequest
        fields = ['model', 'images']


class ModelResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelResponse
        fields = '__all__'


