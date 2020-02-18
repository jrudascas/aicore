from django.contrib import admin

from .models import *

admin.site.register(Model)
admin.site.register(ModelVersion)
admin.site.register(Request)
admin.site.register(RequestType)
admin.site.register(RequestStatus)