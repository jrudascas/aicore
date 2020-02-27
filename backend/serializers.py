from rest_framework import serializers
from .models import *


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'is_staff']


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Model
        fields = '__all__'


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['private_id', 'url_path']
        extra_kwargs = {
            'private_id': {
                'help_text': 'Private ID from the client system'},
            'url': {
                'help_text': "Public URL in which the image is accessible"
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


