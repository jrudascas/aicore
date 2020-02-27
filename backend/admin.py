from django.contrib import admin

from .models import *

admin.site.register(Model)
admin.site.register(ModelRequest)
admin.site.register(ModelResponse)
admin.site.register(ModelRequestType)
admin.site.register(ModelRequestStatus)
admin.site.register(Image)