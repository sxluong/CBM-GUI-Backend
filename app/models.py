from django.db import models

class MachineLearningModel(models.Model):
    model_id = models.CharField(max_length=255, unique=True)  # Ensure model_id is unique
    model_path = models.CharField(max_length=500)  # Path to full model
    fl_weights_path = models.CharField(max_length=500)
    fl_biases_path = models.CharField(max_length=500)
    fl_mean_path = models.CharField(max_length=500)
    fl_std_path = models.CharField(max_length=500)
    backbone = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for model creation 
    