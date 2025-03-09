from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
from .training_scripts import concepts
from .models import MachineLearningModel
from .serializer import MachineLearningModelSerializer
import subprocess
from datetime import datetime
import hashlib
from django.http import JsonResponse
import google.generativeai as genai
from django.conf import settings
import torch
import os
import shutil
import numpy as np
from .training_scripts.prune_model import prune_model
from .training_scripts.evaluate_pruned_model import evaluate_pruned_model
from django.views.decorators.csrf import csrf_exempt
import time

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

        # pruned concepts
        pruned_concepts = request.data.get("pruned_concepts", set())


        # Check if the model already exists
        model = MachineLearningModel.objects.filter(model_id=model_id).first()

        if model:
            concept_set = {'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia}

            concept_set = concept_set[concept_dataset]
            pruned_concepts = set(pruned_concepts)
            new_concept_set = [c for c in concept_set if c not in pruned_concepts]    
            previous_accuracy = model.full_accuracy

            # Simpler pruned model ID with just timestamp and number of pruned concepts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pruned_model_id = f"{model_id}_pruned_{len(pruned_concepts)}_{timestamp}"
            
            # Handle SetFit/sst2 special case
            formatted_dataset = concept_dataset.replace('/', '_') if concept_dataset == 'SetFit/sst2' else concept_dataset
            
            subprocess.run([
                "python",
                "app/training_scripts/get_concept_labels.py",
                f"--dataset={concept_dataset}",  # Keep original format for the script
                "--concept_text_sim_model=mpnet",
                f"--model_id={pruned_model_id}",
                "--custom_concepts"
            ] + new_concept_set, check=True)

            # The actual path structure based on your screenshot
            model_base_path = f"mpnet_acs/{formatted_dataset}/model_{pruned_model_id}/roberta_cbm"
            
            subprocess.run([
                "python", "app/training_scripts/train_CBL.py",
                "--automatic_concept_correction",
                f"--dataset={concept_dataset}",
                f"--backbone={backbone}",
                f"--model_id={pruned_model_id}",
                f"--batch_size=16",
                "--custom_concepts"
            ] + new_concept_set, check=True)

            subprocess.run([
                "python", "app/training_scripts/train_FL.py",
                f"--cbl_path={model_base_path}/cbl_acc_best.pt"
            ], check=True)

            # Read accuracy from the correct path
            accuracy_path = f"{model_base_path}/accuracies.json"
            
            with open(accuracy_path, 'r') as f:
                accuracy_data = json.load(f)
            
            new_accuracy = accuracy_data['full_accuracy']

            # Update the database with the pruned model information
            MachineLearningModel.objects.create(
                model_id=pruned_model_id,
                model_path=f"{model_base_path}/cbl_acc_best.pt",
                fl_weights_path=f"{model_base_path}/W_g_acc_best.pt",
                fl_biases_path=f"{model_base_path}/b_g_acc_best.pt",
                fl_std_path=f"{model_base_path}/train_std_acc_best.pt",
                fl_mean_path=f"{model_base_path}/train_mean_acc_best.pt",
                backbone=backbone,
                full_accuracy=new_accuracy,
                pruned_concepts=list(pruned_concepts),  # Store the pruned concepts
                is_pruned_version=True,
                original_model_id=model_id
            )

            return Response({
                "message": "Model already exists. Retraining logic to be implemented.",
                "previous_accuracy": previous_accuracy,
                "new_accuracy": new_accuracy,
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
                f"--cbl_path=mpnet_acs/{concept_dataset}/model_{model_id}/{backbone}_cbm/cbl_acc_best.pt"
            ], check=True)

            formatted_dataset = concept_dataset.replace('/', '_') if concept_dataset == 'SetFit/sst2' else concept_dataset

            # Save the trained model details
            full_model_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/cbl_acc_best.pt"
            fl_weightings_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/W_g_acc_best.pt"
            fl_biases_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/b_g_acc_best.pt"
            fl_std_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/train_std_acc_best.pt"
            fl_mean_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/train_mean_acc_best.pt"

            # read json to retrieve accuracy
            accuracy_path = f"mpnet_acs/{formatted_dataset}/model_{model_id}/{backbone}_cbm/accuracies.json"

            with open(accuracy_path, 'r') as f:
                accuracy_data = json.load(f)
            
            accuracy = accuracy_data['full_accuracy']
            
            MachineLearningModel.objects.create(
                model_id=model_id,
                model_path=full_model_path,
                fl_weights_path=fl_weightings_path,
                fl_biases_path=fl_biases_path,
                fl_std_path = fl_std_path,
                fl_mean_path = fl_mean_path,
                backbone=backbone,
                full_accuracy=accuracy,
            )

            return Response({
                "message": "Model processing complete",
                "model_path": full_model_path,
                "model_accuracy": accuracy,
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

        try:
            # Look up the model object in your database
            model_obj = MachineLearningModel.objects.get(model_id=model_id)

            # Get the direct paths to the required model files
            cbl_model_path = model_obj.model_path
            train_mean_path = model_obj.fl_mean_path
            train_std_path = model_obj.fl_std_path
            fl_w_path = model_obj.fl_weights_path
            fl_b_path = model_obj.fl_biases_path
            backbone = model_obj.backbone
            pruned_concepts = model_obj.pruned_concepts or []
            is_pruned_version = model_obj.is_pruned_version or False
            
            print(f"cbl_model_path: {cbl_model_path}")
            
            # If concept_dataset is not provided, extract just the dataset name from the path
            if not concept_dataset:
                # The dataset name should be the second part of the path
                # e.g., from "mpnet_acs/SetFit_sst2/..." we want "SetFit/sst2"
                dataset_part = cbl_model_path.split('/')[1]
                if dataset_part == "SetFit_sst2":
                    concept_dataset = "SetFit/sst2"
                else:
                    concept_dataset = dataset_part
            
            print(f"Using concept_dataset: {concept_dataset}")

            # Fix paths for file system
            cbl_model_path = cbl_model_path.replace("SetFit/sst2", "SetFit_sst2")
            train_mean_path = train_mean_path.replace("SetFit/sst2", "SetFit_sst2")
            train_std_path = train_std_path.replace("SetFit/sst2", "SetFit_sst2")
            fl_w_path = fl_w_path.replace("SetFit/sst2", "SetFit_sst2")
            fl_b_path = fl_b_path.replace("SetFit/sst2", "SetFit_sst2")

            # Print the command we're about to run
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
                "--include_contributions", "true"
            ]
            print("Running command:", " ".join(cmd))

            # Run the script and capture output
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )

            print("Script stdout:", result.stdout)
            print("Script stderr:", result.stderr)
            print("Return code:", result.returncode)

            if result.returncode != 0:
                return JsonResponse({
                    "error": f"Script failed with error: {result.stderr}",
                    "stdout": result.stdout,
                    "returncode": result.returncode
                }, status=500)

            try:
                output = json.loads(result.stdout)
                
                # Add a flag for frontend to know whether to filter concepts
                output["pruned_concepts"] = pruned_concepts
                output["is_pruned_model"] = is_pruned_version
                
                # REMOVED: The code that zero'd out activations for pruned concepts
                # Just return the output as is
                return JsonResponse(output)
                
            except json.JSONDecodeError as e:
                return JsonResponse({
                    "error": "Failed to parse JSON output",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "parse_error": str(e)
                }, status=500)

        except Exception as e:
            import traceback
            return JsonResponse({
                "error": f"Server error: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)
        

class BiasDetectionView(APIView):
    """
    View for detecting biased concepts using Gemini API.
    """
    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def options(self, request, *args, **kwargs):
        response = Response()
        response["Access-Control-Allow-Origin"] = "http://localhost:3050"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def post(self, request):
        try:
            # Get the classification report from the request
            classification_data = request.data.get('classification_data')
            print("Received classification data:", classification_data)
            
            if not classification_data:
                print("No classification data provided")
                return JsonResponse({
                    "error": "No classification data provided"
                }, status=400)

            # Configure Gemini
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash-001')

            # Print the prompt for debugging
            prompt = f"""
            Analyze the following concept classification report and identify at least 3 potentially biased concepts.
            Focus on concepts that might reflect social, cultural, gender, racial, or other forms of bias, even if subtle.
            Consider both explicit and implicit biases in the concepts.

            Classification Report:
            {json.dumps(classification_data, indent=2)}

            Return your analysis as a JSON object with this exact structure, without any markdown formatting or code blocks:
            {{
                "biased_concepts": [
                    {{
                        "concept": "concept_name",
                        "bias_type": "type of bias (e.g., gender, cultural, social, racial)",
                        "reasoning": "detailed explanation of why this concept might be biased",
                        "severity": "high/medium/low",
                        "activation": float_value
                    }}
                ]
            }}

            Important:
            1. Identify at least 3 concepts that show any signs of potential bias
            2. Be thorough in explaining the reasoning
            3. Consider subtle and implicit biases
            4. Do not wrap the response in markdown code blocks
            """
            print("Sending prompt to Gemini:", prompt)

            # Get response from Gemini
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '', 1)
            if response_text.endswith('```'):
                response_text = response_text.replace('```', '', 1)
            
            response_text = response_text.strip()
            print("Cleaned response:", response_text)
            
            try:
                # Parse the response to ensure it's valid JSON
                bias_analysis = json.loads(response_text)
                print("Parsed bias analysis:", bias_analysis)
                return JsonResponse(bias_analysis)
            except json.JSONDecodeError as e:
                print("JSON parsing error:", str(e))
                return JsonResponse({
                    "error": "Failed to parse Gemini response as JSON",
                    "raw_response": response_text
                }, status=500)

        except Exception as e:
            print("Error in bias detection:", str(e))
            return JsonResponse({
                "error": f"Error analyzing bias: {str(e)}"
            }, status=500)
        

class ConceptPruningView(APIView):
    """
    View for pruning concepts by setting their weights to zero.
    """
    def dispatch(self, request, *args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] === ConceptPruningView dispatch called ===")
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
        try:
            print(f"[{time.strftime('%H:%M:%S')}] === ConceptPruningView POST started ===")
            data = json.loads(request.body)
            model_id = data.get('model_id')
            concepts_to_prune = data.get('concepts_to_prune', [])
            concept_indices = data.get('concept_indices', [])
            use_indices = data.get('use_indices', False)
            
            print(f"[{time.strftime('%H:%M:%S')}] Pruning request - Model: {model_id}, Concepts: {concepts_to_prune}")
            
            # Check for required parameters
            if not model_id:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: Missing model_id")
                return JsonResponse({"success": False, "error": "Model ID is required"}, status=400)
            
            if not concepts_to_prune and not concept_indices:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: No concepts to prune specified")
                return JsonResponse({"success": False, "error": "No concepts to prune specified"}, status=400)
            
            # Get the original model
            original_model = MachineLearningModel.objects.filter(model_id=model_id).first()
            if not original_model:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: Model {model_id} not found")
                return JsonResponse({"success": False, "error": f"Model {model_id} not found"}, status=404)
            
            # Extract dataset from the model path
            print(f"[{time.strftime('%H:%M:%S')}] Extracting dataset from model path")
            model_path_parts = original_model.model_path.split('/')
            if len(model_path_parts) >= 2:
                formatted_dataset = model_path_parts[1]  # e.g., SetFit_sst2
                concept_dataset = formatted_dataset.replace('_', '/') if formatted_dataset == 'SetFit_sst2' else formatted_dataset
                print(f"[{time.strftime('%H:%M:%S')}] Extracted dataset: {concept_dataset}")
            else:
                concept_dataset = "SetFit/sst2"  # default
                print(f"[{time.strftime('%H:%M:%S')}] Using default dataset: {concept_dataset}")
            
            print(f"[{time.strftime('%H:%M:%S')}] Calling prune_model function...")
            
            # Call the prune_model function with the CORRECT parameter name
            pruning_result = prune_model(
                model_id=model_id,
                pruned_concepts=concepts_to_prune,
                concept_dataset=concept_dataset,
                backbone=original_model.backbone
            )
            
            print(f"[{time.strftime('%H:%M:%S')}] Pruning completed. Result: {pruning_result}")
            
            # Use the pruned_model_id returned by prune_model()
            pruned_model_id = pruning_result["pruned_model_id"]
            print(f"[{time.strftime('%H:%M:%S')}] Using pruned model ID from function: {pruned_model_id}")
            
            # Create a new model record
            new_model = MachineLearningModel.objects.create(
                model_id=pruned_model_id,
                model_path=pruning_result["model_path"],
                fl_mean_path=pruning_result["fl_mean_path"],
                fl_std_path=pruning_result["fl_std_path"],
                fl_weights_path=pruning_result["fl_weights_path"],
                fl_biases_path=pruning_result["fl_biases_path"],
                backbone=original_model.backbone,
                full_accuracy=pruning_result["original_accuracy"],
                pruned_concepts=concepts_to_prune,
                is_pruned_version=True,
                original_model_id=model_id
            )
            
            print(f"[{time.strftime('%H:%M:%S')}] Created new pruned model: {pruned_model_id}")
            
            # Start evaluation immediately
            print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation of pruned model...")
            try:
                eval_result = evaluate_pruned_model(
                    model_id=pruned_model_id,
                    concept_dataset=concept_dataset,
                    backbone=original_model.backbone
                )
                # Include evaluation results in the response
                return JsonResponse({
                    "success": True,
                    "message": "Concepts pruned and evaluated successfully",
                    "pruned_model_id": pruned_model_id,
                    "pruned_concepts": concepts_to_prune,
                    "original_accuracy": pruning_result["original_accuracy"],
                    "pruned_accuracy": eval_result["accuracy"],
                    "accuracy_change": eval_result["accuracy"] - pruning_result["original_accuracy"],
                    "evaluation_completed": True
                })
            except Exception as eval_error:
                print(f"[{time.strftime('%H:%M:%S')}] Evaluation failed: {str(eval_error)}")
                # Return success for pruning but note that evaluation failed
                return JsonResponse({
                    "success": True,
                    "message": "Concepts pruned successfully, but evaluation failed",
                    "pruned_model_id": pruned_model_id,
                    "pruned_concepts": concepts_to_prune,
                    "original_accuracy": pruning_result["original_accuracy"],
                    "evaluation_completed": False,
                    "evaluation_error": str(eval_error)
                })
            
        except Exception as e:
            import traceback
            print(f"[{time.strftime('%H:%M:%S')}] ERROR during pruning: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                "success": False,
                "error": f"Error processing request: {str(e)}",
                "traceback": traceback.format_exc()
            }, status=500)
        

@csrf_exempt
def evaluate_model(request):
    # Add CORS headers to all responses
    response_headers = {
        "Access-Control-Allow-Origin": "http://localhost:3050",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    
    # Handle preflight OPTIONS requests for CORS
    if request.method == 'OPTIONS':
        response = JsonResponse({})
        for key, value in response_headers.items():
            response[key] = value
        return response
    
    if request.method == 'POST':
        try:
            print(f"[{time.strftime('%H:%M:%S')}] === evaluate_model endpoint called ===")
            data = json.loads(request.body)
            model_id = data.get('model_id')
            dataset = data.get('dataset', 'SetFit/sst2')
            backbone = data.get('backbone', 'roberta')
            
            print(f"[{time.strftime('%H:%M:%S')}] Evaluating model {model_id} on {dataset} with {backbone} backbone")
            
            if not model_id:
                print(f"[{time.strftime('%H:%M:%S')}] ERROR: Model ID is required")
                response = JsonResponse({'success': False, 'error': 'Model ID is required'}, status=400)
            else:
                # Call the evaluation function
                print(f"[{time.strftime('%H:%M:%S')}] Starting evaluation...")
                result = evaluate_pruned_model(
                    model_id=model_id,
                    concept_dataset=dataset,
                    backbone=backbone
                )
                print(f"[{time.strftime('%H:%M:%S')}] Evaluation completed: Accuracy = {result['accuracy']}")
                
                response = JsonResponse({
                    'success': True,
                    'accuracy': result['accuracy'],
                    'pruned_concepts': result['pruned_concepts'],
                    'total_samples': result['total_samples'],
                    'correct_predictions': result['correct_predictions']
                })
        except Exception as e:
            import traceback
            print(f"[{time.strftime('%H:%M:%S')}] ERROR during evaluation: {str(e)}")
            print(traceback.format_exc())
            response = JsonResponse({'success': False, 'error': str(e)}, status=500)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Method {request.method} not allowed")
        response = JsonResponse({'success': False, 'error': 'Only POST method is allowed'}, status=405)
    
    # Add CORS headers to the response
    for key, value in response_headers.items():
        response[key] = value
    
    return response
        
