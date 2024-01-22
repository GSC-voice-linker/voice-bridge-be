# 전체 프로젝트의 메인 URL 설정을 담당

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('stt/', include("stt.urls")), # 'stt/' URL 경로로 들어오는 요청을 'stt.urls' 파일로 전달.
    path('admin/', admin.site.urls),
]
