from django.contrib import admin
from django.urls import path

from spinefracture import views

urlpatterns = [
    path('admin/', admin.site.urls),
    #
    path("spinefracture/", views.call_model.as_view())
]
