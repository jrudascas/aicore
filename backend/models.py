from django.db import models
from django.contrib.auth.models import User


class Model(models.Model):
    name = models.CharField(max_length=200)
    model_factory_name = models.CharField(max_length=200, null=True)

    def __str__(self):
        return self.name


class ModelRequestType(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class ModelRequestStatus(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class ModelRequest(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    model = models.ForeignKey(Model, on_delete=models.CASCADE, null=True)

    type = models.ForeignKey(ModelRequestType, on_delete=models.CASCADE)
    status = models.ForeignKey(ModelRequestStatus, on_delete=models.CASCADE)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.id) + ' ' + self.user.__str__() + ' --> ' + self.model.__str__() + ' : ' + self.created_at.__str__()


class ModelResponse(models.Model):
    request = models.ForeignKey(ModelRequest, on_delete=models.CASCADE, null=True)
    response = models.CharField(null=True, max_length=20000)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.id) + ' ' + self.request.__str__() + ' : ' + self.created_at.__str__()