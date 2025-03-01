from rest_framework import serializers
from .models import MachineLearningModel

class MachineLearningModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MachineLearningModel
        fields = [
            'model_id',
            'model_path',
            'fl_weights_path',
            'fl_biases_path',
            'fl_mean_path',
            'fl_std_path',
            'backbone',
            'created_at'
            'full_accuracy'
        ]