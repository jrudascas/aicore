from django.urls import path
from django.contrib import admin
from django.conf.urls import url, include
from rest_framework import routers
from backend.views import *
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
   openapi.Info(
      title="ImexHS - AI Core API",
      default_version='v1',
      description="This is the official documentation and technical manual to use the AI Core API from ImexHS",
      terms_of_service="https://www.imexhs.com/aicore/",
      contact=openapi.Contact(email="jorge.rudas@imexhs.com"),
      license=openapi.License(name="Only for internal use"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)

router = routers.DefaultRouter()
router.register(r'user', UserViewSet)
router.register(r'image', ImageViewSet)
router.register(r'model', ModelViewSet)
router.register(r'request', ModelRequestViewSet)
router.register(r'response', ModelResponseViewSet)


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls')),
    url(r'^doc/$', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
]
