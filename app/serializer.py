from rest_framework import serializers
from .models import MachineLearningModel

class MachineLearningModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MachineLearningModel
        fields = ['model_id', 'model_path', 'sparse_model_path', 'created_at']