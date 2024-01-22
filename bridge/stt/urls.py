from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"), # URL 패턴에 index라는 이름을 부여하여 참조하기 쉽게 함.
]