from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('disease/', views.disease, name='disease'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('result/', views.result, name='result'),
    path('about/', views.about, name='about'), 
    path('contact/', views.contact, name='contact'), 

]
