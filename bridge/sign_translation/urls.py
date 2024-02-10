from django.urls import path
from . import views

urlpatterns = [
    path("", views.sign_translation, name='sign_translation'),
]