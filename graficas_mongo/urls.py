from django.contrib import admin
from django.urls import path
from visualizaciones.views import graficas

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', graficas, name='graficas'),
]
