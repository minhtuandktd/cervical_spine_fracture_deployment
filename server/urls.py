from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings
from django.views.static import serve

from spinefracture import views

urlpatterns = [
    path('admin/', admin.site.urls),
    #
    path("spinefracture/", views.call_model.as_view()),
    re_path(r'^api/v1/uploads/(?P<path>.*)$', serve,{'document_root': settings.UPLOADS_ROOT}),
    re_path(r'^api/v1/static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]
