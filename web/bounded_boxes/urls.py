from django.conf.urls import url

from . import views

app_name = 'bounded_boxes'
urlpatterns = [
    url(r'^$', views.index_view, name='index'),
    url(r'^export/$', views.export_training_data_view, name='export_training_data'),
]