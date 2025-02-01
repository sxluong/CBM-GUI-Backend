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
            return Response(None, status=status.HTTP_200_OK)  # Return None if model_id is missing
        
        model = MachineLearningModel.objects.filter(model_id=model_id).first()
        if model:
            # If the model exists, return its serialized data
            serializer = MachineLearningModelSerializer(model)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            # If the model does not exist, return None
            return Response(None, status=status.HTTP_200_OK)

    def post(self, request):
        """
        POST endpoint: Train a model if it does not already exist.
        """
        # All these should be strings
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
            subprocess.run([
                "python",
                "training_scripts/get_concept_labels.py",
                f"--dataset={concept_dataset}",
                "--concept_text_sim_model=mpnet",
                f"--model_id={model_id}"
            ], check=True)

            subprocess.run([
                "python", "training_scripts/train_CBL.py",
                "--automatic_concept_correction",
                f"--dataset={concept_dataset}",
                f"--backbone={backbone}",
                f"--model_id={model_id}",
                f"--batch_size=16"
            ], check=True)

            subprocess.run([
                "python", "train_FL.py",
                f"--cbl_path=mpnet_acs/{concept_dataset}/{backbone}_cbm/model_{model_id}/cbl_acc_epoch_8.pt",
                f"--backbone={backbone}"
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
                "message": "Model processing complete",
                "model_path": full_model_path,
                "sparse_model_path": sparse_model_path
            }, status=status.HTTP_200_OK)