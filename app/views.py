from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import MachineLearningModel
import subprocess

class TrainModelView(APIView):
    def post(self, request):
        # Parse request data
        concept_dataset = request.data.get("concept_dataset", None)
        model_type = request.data.get("model_type") # can be LLM or Computer Vision(Maybe Later)
        backbone = request.data.get("backbone") # either roberta or gpt2
        model_id = request.data.get("model_id") # Passed in from the tab
        hardware = request.data.get("hardware", "Local Hardware")  # Default hardware
        
        model = MachineLearningModel.objects.filter(model_id=model_id).first()
        if model:
            #retraining here
            pass
        else: 
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

            # save the model now
            full_model_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/W_g.pt"
            sparse_model_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/W_g_sparse.pt"

            # Create and save the model in Django
            MachineLearningModel.objects.create(
                model_id=model_id,
                model_path=full_model_path,
                sparse_model_path=sparse_model_path
            )


            return Response({"message": "Model processing complete"}, status=200)
