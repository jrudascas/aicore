from django.db import models
from django.contrib.auth.models import User


class ModelVersion(models.Model):
    number = models.IntegerField(default=0)

    def __str__(self):
        return self.number


class Model(models.Model):
    name = models.CharField(max_length=200)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class RequestType(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class RequestStatus(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class Request(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    date = models.DateField(null=True)
    type = models.ForeignKey(RequestType, on_delete=models.CASCADE)
    status = models.ForeignKey(RequestStatus, on_delete=models.CASCADE)

    def __str__(self):
        return self.id + str(self.date)