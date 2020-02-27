from django.db import models
from django.contrib.auth.models import User
import uuid


ID_FIELD_LENGTH = 36


class Model(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    name = models.CharField(max_length=200)
    model_factory_name = models.CharField(max_length=200, null=True)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid + ' --> ' + self.name


class ModelRequestType(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    name = models.CharField(max_length=200)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid + ' --> ' + self.name


class ModelRequestStatus(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    name = models.CharField(max_length=200)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid + ' --> ' + self.name


class ModelRequest(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
    type = models.ForeignKey(ModelRequestType, on_delete=models.CASCADE)
    status = models.ForeignKey(ModelRequestStatus, on_delete=models.CASCADE)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid + ' --> ' + self.user.__str__() + ' - ' + self.model.__str__()


class ModelResponse(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    request = models.ForeignKey(ModelRequest, on_delete=models.CASCADE, null=True)
    response = models.CharField(null=True, max_length=20000)

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid + ' --> ' + self.request.__str__() + ' : ' + self.created_at.__str__()


class Image(models.Model):
    uuid = models.CharField(primary_key=True, max_length=ID_FIELD_LENGTH)
    url_path = models.CharField(max_length=256, null=True)
    #array = models.CharField(max_length=2000, null=True)
    request = models.ForeignKey(ModelRequest, related_name='images', on_delete=models.CASCADE)
    private_id = models.CharField(max_length=20, null=True)

    def save(self, *args, **kwargs):
        if not self.uuid:
            self.uuid = uuid.uuid1()
        super().save(*args, **kwargs)

    def __str__(self):
        return self.uuid

