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
    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        # Add CORS headers
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def options(self, request, *args, **kwargs):
        # Handle preflight requests
        response = Response()
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
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
                "app/training_scripts/get_concept_labels.py",
                f"--dataset={concept_dataset}",
                "--concept_text_sim_model=mpnet",
                f"--model_id={model_id}"
            ], check=True)

            subprocess.run([
                "python", "app/training_scripts/train_CBL.py",
                "--automatic_concept_correction",
                f"--dataset={concept_dataset}",
                f"--backbone={backbone}",
                f"--model_id={model_id}",
                f"--batch_size=16"
            ], check=True)

            subprocess.run([
                "python", "app/training_scripts/train_FL.py",
                f"--cbl_path=mpnet_acs/{concept_dataset}/{backbone}_cbm/model_{model_id}/cbl_acc_best.pt",
                f"--backbone={backbone}"
            ], check=True)

            # Save the trained model details
            full_model_path = f"mpnet_acs/{concept_dataset}/{backbone}_cbm/model_{model_id}/cbl_acc_best.pt"
            fl_weightings_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/W_g_acc_best.pt"
            fl_biases_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/b_g_acc_best.pt"
            fl_std_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/train_std_acc_best.pt"
            fl_mean_path = f"mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/train_mean_acc_best.pt"
            MachineLearningModel.objects.create(
                model_id=model_id,
                model_path=full_model_path,
                fl_weightings_path=fl_weightings_path,
                fl_biases_path=fl_biases_path,
                fl_std_path = fl_std_path,
                fl_mean_path = fl_mean_path,
                backbone=backbone,
            )

            return Response({
                "message": "Model processing complete",
                "model_path": full_model_path,
            }, status=status.HTTP_200_OK)
        

class ClassificationView(APIView):
    """
    View for handling Classification
    """
    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        # Add CORS headers
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def options(self, request, *args, **kwargs):
        # Handle preflight requests
        response = Response()
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    

def post(self, request):
    model_id = request.data.get("model_id")
    concept_dataset = request.data.get("concept_dataset")
    input_text = request.data.get("input")

    # Look up the model object in your database
    model_obj = MachineLearningModel.objects.get(model_id=model_id)

    # Get the direct paths to the required model files
    cbl_model_path = model_obj.model_path  # Adjust attribute name if necessary
    train_mean_path = model_obj.fl_mean_path
    train_std_path = model_obj.fl_std_path
    fl_w_path = model_obj.fl_weights_path
    fl_b_path = model_obj.fl_biases_path
    backbone = model_obj.backbone  # "roberta" or "gpt2"

    # Build the command to call the script
    cmd = [
        "python", "app/training_scripts/test_model.py",
        "--input", input_text,
        "--backbone", backbone,
        "--concept_dataset", concept_dataset,
        "--cbl_model_path", cbl_model_path,
        "--train_mean_path", train_mean_path,
        "--train_std_path", train_std_path,
        "--fl_w_path", fl_w_path,
        "--fl_b_path", fl_b_path,
    ]

    # Optionally, add the dropout parameter if needed
    # cmd.extend(["--dropout", "0.1"])

    # Use Popen to run the subprocess and capture its output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        return JsonResponse({"error": stderr}, status=500)

    try:
        output = json.loads(stdout)
    except json.JSONDecodeError as e:
        return JsonResponse({"error": f"JSON decoding error: {e}"}, status=500)

    return JsonResponse(output)
        





        


