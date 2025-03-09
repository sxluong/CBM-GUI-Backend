"""
URL configuration for project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from app.views import MachineLearningModelView, ClassificationView, BiasDetectionView, ConceptPruningView, evaluate_model

urlpatterns = [
    path('admin/', admin.site.urls),
    path('process-model/', MachineLearningModelView.as_view(), name='process-model'), 
    path('classify-model/', ClassificationView.as_view(), name='classify-model'),
    path('detect-bias/', BiasDetectionView.as_view(), name='detect-bias'),
    path('prune-concepts/', ConceptPruningView.as_view(), name='prune-concepts'),
    path('evaluate-model/', evaluate_model, name='evaluate_model'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
