from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import MachineLearningModel
from .serializer import MachineLearningModelSerializer
import subprocess

class MachineLearningModelView(APIView):
    """
    View for handling Machine Learning Model retrieval and training.
    """

    def get(self, request):
        """
        GET endpoint: Check if the model exists based on the model_id and return its details if found.
        """
        model_id = request.query_params.get("model_id")  # Extract model_id from query parameters
        
        if not model_id:
            return Response({"error": "model_id is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        model = MachineLearningModel.objects.filter(model_id=model_id).first()
        if model:
            # If the model exists, return its details
            serializer = MachineLearningModelSerializer(model)
            return Response({
                "exists": True,
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        else:
            # If the model doesn't exist, return `exists: False`
            return Response({
                "exists": False,
                "message": "Model not found"
            }, status=status.HTTP_200_OK)

    def post(self, request):
        """
        POST endpoint: Train a model if it does not already exist.
        """
        # Parse request data
        concept_dataset = request.data.get("concept_dataset", None)
        model_type = request.data.get("model_type")  # Can be LLM or Computer Vision (future use)
        backbone = request.data.get("backbone")  # Either roberta or gpt2
        model_id = request.data.get("model_id")  # Passed in from the frontend tab
        hardware = request.data.get("hardware", "Local Hardware")  # Default hardware

        # Check if the model already exists
        model = MachineLearningModel.objects.filter(model_id=model_id).first()
        if model:
            # If the model exists, handle retraining logic here
            return Response({
                "message": "Model already exists. Retraining logic to be implemented."
            }, status=status.HTTP_200_OK)
        else:
            # Train the model using subprocess calls
            subprocess.run([
                "python", "get_concept_labels.py",
                "--dataset", concept_dataset,
                "--concept_text_sim_model", "mpnet",
                "--model_id", str(model_id)
            ], check=True)

            subprocess.run([
                "python", "training_scripts/train_CBL.py",
                "--automatic_concept_correction",
                "--dataset", concept_dataset,
                "--backbone", backbone,
            ], check=True)

            subprocess.run([
                "python", "train_FL.py", 
                "--cbl_path", f"mpnet_acs/{concept_dataset}/{backbone}_cbm/model_{model_id}/cbl_acc_epoch_8.pt",
                "--backbone", backbone,
            ], check=True)

            # Save the trained model details
            full_model_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/W_g.pt"
            sparse_model_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/W_g_sparse.pt"

            MachineLearningModel.objects.create(
                model_id=model_id,
                model_path=full_model_path,
                sparse_model_path=sparse_model_path
            )

            return Response({
                "message": "Model processing complete"
            }, status=status.HTTP_200_OK)