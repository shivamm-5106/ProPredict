from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import esm
import pickle
import os
import numpy as np
app = FastAPI(title="Protein Analysis Pipeline")
# CORS middleware to allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD GO TERMS ====================
from go_parser import parse_obo_file, create_go_index_mapping

go_terms = parse_obo_file("./go-basic.obo")
GO_INDEX_MAPPING, GO_ID_TO_INDEX = create_go_index_mapping(go_terms)

try:
    print(f"📖 Loading GO terms from {GO_OBO_PATH} ...")
    go_terms = parse_obo_file(GO_OBO_PATH)
    GO_INDEX_MAPPING, GO_ID_TO_INDEX = create_go_index_mapping(go_terms)
    print(f"✅ Loaded {len(GO_INDEX_MAPPING)} GO terms successfully")
except FileNotFoundError:
    print(f"⚠️ GO OBO file not found at {GO_OBO_PATH}. Skipping GO integration.")
except Exception as e:
    print(f"⚠️ Failed to parse GO OBO file: {e}")


# ==================== MODEL 1: ESM2 Embedding ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading ESM2-150M model...")
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()
print("Model loaded successfully!")
EMB_PATH = "embeddings_150M.pkl"
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "rb") as f:
        embedding_dict = pickle.load(f)
    print(f"✅ Loaded {len(embedding_dict)} cached embeddings")
else:
    embedding_dict = {}
    print("⚠️ No existing embeddings found")
def get_embedding(sequence: str, seq_id: str):
    if seq_id in embedding_dict:
        print(f"ℹ️ Found cached embedding for {seq_id}")
        return embedding_dict[seq_id]

    # Prepare batch
    data = [(seq_id, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Generate embedding
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30])
        token_representations = results["representations"][30]

    # Mean pooling
    embedding = token_representations.mean(1).squeeze().cpu()

    # Save to cache
    embedding_dict[seq_id] = embedding
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embedding_dict, f)

    print(f"✅ Generated and cached embedding for {seq_id}")
    
    return embedding
# ==================== MODEL 2: Secondary Structure Prediction ====================

MODEL2_PATH = "./models/Model_1.pt"  # Change this to your model path
# Define your model architecture (must match training architecture)
class Model2Network(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, output_dim),
            torch.nn.Sigmoid()  # multi-label
        )
    def forward(self, x):
        return self.layers(x)
# Load pretrained model
model2 = None
if os.path.exists(MODEL2_PATH):
    print(f"Loading Model 2 from {MODEL2_PATH}...")
    model2 = Model2Network(input_dim=640, output_dim=26121)  # Adjust dimensions as needed
    model2.load_state_dict(torch.load(MODEL2_PATH, map_location=device))
    model2 = model2.to(device)
    model2.eval()
    print("✅ Model 2 loaded successfully!")
else:
    print(f"⚠️ Model 2 not found at {MODEL2_PATH} - using placeholder")
def predict_structure_from_embedding(embedding: torch.Tensor):
    """
    Model 2: Takes 640-dim embedding and returns structure predictions
    
    Args:
        embedding (torch.Tensor): Shape (640,) - output from get_embedding()
    
    Returns:
        dict: {
            "predictions": torch.Tensor or numpy array of shape (3,),
            "probabilities": {
                "alpha_helix": float (0-100),
                "beta_sheet": float (0-100), 
                "coil": float (0-100)
            },
            "raw_output": list of 3 probabilities,
            "model_loaded": bool
        }
    """
    if model2 is None:
        print("⚠️ Model 2 not loaded - cannot make predictions")
        return {
            "predictions": None,
            "probabilities": {
                "alpha_helix": 0.0,
                "beta_sheet": 0.0,
                "coil": 0.0
            },
            "raw_output": [0.0, 0.0, 0.0],
            "model_loaded": False,
            "error": "Model 2 not loaded"
        }
    
    # Ensure embedding is on correct device and shape
    with torch.no_grad():
        # Input should be [1, 640] for batch processing
        if embedding.dim() == 1:
            embedding_input = embedding.unsqueeze(0)  # [640] -> [1, 640]
        else:
            embedding_input = embedding
        
        embedding_input = embedding_input.to(device)
        
        # Forward pass through Model 2
        output = model2(embedding_input)  # Shape: [1, 3]
        output = output.cpu().squeeze()    # Shape: [3]
        
        # Convert to numpy for easier handling
        predictions = output.numpy()
    
    # Convert to percentages
    probabilities = {
        "alpha_helix": float(predictions[0] * 100),
        "beta_sheet": float(predictions[1] * 100),
        "coil": float(predictions[2] * 100)
    }
    
    return {
        "predictions": predictions,  # numpy array [3]
        "probabilities": probabilities,
        "raw_output": predictions.tolist(),
        "model_loaded": True
    }
def predict_structure(embedding: torch.Tensor, sequence: str):
    """
    Model 2: Predict using pretrained neural network (API wrapper)
    This function wraps predict_structure_from_embedding for API compatibility
    """
    result = predict_structure_from_embedding(embedding)
    
    if not result["model_loaded"]:
        # Fallback placeholder if model not loaded
        embedding_np = embedding.numpy()
        mean_val = np.mean(embedding_np)
        std_val = np.std(embedding_np)
        
        helix_percent = min(max(15 + (mean_val * 50), 10), 60)
        sheet_percent = min(max(20 + (std_val * 30), 10), 50)
        coil_percent = 100 - helix_percent - sheet_percent
        
        return {
            "alpha_helix": round(helix_percent, 2),
            "beta_sheet": round(sheet_percent, 2),
            "coil": round(coil_percent, 2),
            "sequence_length": len(sequence),
            "model_status": "placeholder"
        }
    
    return {
        "alpha_helix": round(result["probabilities"]["alpha_helix"], 2),
        "beta_sheet": round(result["probabilities"]["beta_sheet"], 2),
        "coil": round(result["probabilities"]["coil"], 2),
        "sequence_length": len(sequence),
        "model_status": "pretrained",
        "raw_predictions": result["raw_output"]
    }
# ==================== MODEL 3: Multilabel Classifier (PKL) ====================
MODEL3_PATH = "./models/Model_2.pkl"  # Change this to your pkl file path
# Load pretrained Model 3 (pkl file)
model3_classifier = None
if os.path.exists(MODEL3_PATH):
    print(f"Loading Model 3 from {MODEL3_PATH}...")
    with open(MODEL3_PATH, "rb") as f:
        model3_classifier = pickle.load(f)
    print("✅ Model 3 loaded successfully!")
else:
    print(f"⚠️ Model 3 not found at {MODEL3_PATH} - using placeholder")
# Function labels (adjust based on your actual labels)
FUNCTION_LABELS = [
    "Enzyme",
    "Structural Protein",
    "Transport Protein",
    "Binding Protein",
    "Signaling Protein",
    "Immune Function"
]
def predict_function_from_model2(model2_output: np.ndarray, threshold=0.5):
    """
    Model 3: Multilabel classifier taking Model 2 output
    
    Args:
        model2_output (np.ndarray): Shape (26121,) - output from Model 2
        threshold (float): Threshold for binary classification (default: 0.5)
    
    Returns:
        dict: {
            "labels": list of predicted function labels,
            "probabilities": dict mapping each label to probability,
            "binary_predictions": dict mapping each label to 0/1,
            "raw_output": model output,
            "model_loaded": bool
        }
    """
    if model3_classifier is None:
        print("⚠️ Model 3 not loaded - using placeholder predictions")
        # Placeholder: random predictions
        placeholder_probs = np.random.rand(len(FUNCTION_LABELS))
        prob_dict = {FUNCTION_LABELS[i]: float(placeholder_probs[i]) for i in range(len(FUNCTION_LABELS))}
        binary_dict = {label: int(prob > threshold) for label, prob in prob_dict.items()}
        predicted_labels = [label for label, binary in binary_dict.items() if binary == 1]
        
        return {
            "labels": predicted_labels,
            "probabilities": prob_dict,
            "binary_predictions": binary_dict,
            "raw_output": placeholder_probs.tolist(),
            "model_loaded": False,
            "error": "Model 3 not loaded"
        }
    
    try:
        # Reshape input for sklearn models
        if len(model2_output.shape) == 1:
            input_data = model2_output.reshape(1, -1)
        else:
            input_data = model2_output
        
        # Predict using the loaded pkl model
        # This handles different sklearn model types automatically
        if hasattr(model3_classifier, 'predict_proba'):
            # Classifier with probability output
            probabilities_array = model3_classifier.predict_proba(input_data)[0]
        elif hasattr(model3_classifier, 'decision_function'):
            # SVM or similar
            decision_scores = model3_classifier.decision_function(input_data)[0]
            # Convert to probabilities using sigmoid
            probabilities_array = 1 / (1 + np.exp(-decision_scores))
        else:
            # Direct prediction
            predictions = model3_classifier.predict(input_data)[0]
            probabilities_array = predictions
        
        # Apply threshold for binary predictions
        binary_preds = (probabilities_array >= threshold).astype(int)
        
        # Map to labels (adjust based on your model's output)
        # If your model outputs specific labels, use those
        if len(probabilities_array) == len(FUNCTION_LABELS):
            prob_dict = {FUNCTION_LABELS[i]: float(probabilities_array[i]) for i in range(len(FUNCTION_LABELS))}
            binary_dict = {FUNCTION_LABELS[i]: int(binary_preds[i]) for i in range(len(FUNCTION_LABELS))}
        else:
            # If output size doesn't match, use generic labels
            prob_dict = {f"Function_{i}": float(probabilities_array[i]) for i in range(len(probabilities_array))}
            binary_dict = {f"Function_{i}": int(binary_preds[i]) for i in range(len(probabilities_array))}
        
        predicted_labels = [label for label, binary in binary_dict.items() if binary == 1]
        
        return {
            "labels": predicted_labels,
            "probabilities": prob_dict,
            "binary_predictions": binary_dict,
            "raw_output": probabilities_array.tolist(),
            "model_loaded": True,
            "threshold": threshold
        }
        
    except Exception as e:
        print(f"❌ Error in Model 3 prediction: {str(e)}")
        return {
            "labels": [],
            "probabilities": {},
            "binary_predictions": {},
            "raw_output": [],
            "model_loaded": True,
            "error": str(e)
        }
# ==================== COMPLETE PIPELINE ====================
def run_complete_pipeline(sequence: str, seq_id: str, threshold=0.5):
    """
    Complete 3-model pipeline:
    Model 1 (ESM2) → Model 2 (NN) → Model 3 (PKL Classifier)
    
    Args:
        sequence (str): Protein amino acid sequence
        seq_id (str): Sequence identifier
        threshold (float): Threshold for Model 3 multilabel classification
    
    Returns:
        dict: Complete pipeline results from all 3 models
    """
    print(f"\n{'='*60}")
    print(f"🧬 Running Complete Pipeline for {seq_id}")
    print(f"{'='*60}")
    
    # ===== MODEL 1: Generate Embedding =====
    print("\n[MODEL 1] Generating ESM2 embedding...")
    embedding = get_embedding(sequence, seq_id)
    print(f"✓ Embedding shape: {embedding.shape}")
    
    model1_result = {
        "embedding": embedding,
        "shape": embedding.shape,
        "seq_id": seq_id
    }
    
    # ===== MODEL 2: Predict Structure =====
    print("\n[MODEL 2] Running neural network prediction...")
    model2_result = predict_structure_from_embedding(embedding)
    
    if not model2_result["model_loaded"]:
        print("⚠️ Model 2 not loaded - pipeline cannot continue")
        return {
            "model1": model1_result,
            "model2": model2_result,
            "model3": None,
            "error": "Model 2 not loaded"
        }
    
    model2_predictions = model2_result["predictions"]  # numpy array shape (26121,)
    print(f"✓ Model 2 output shape: {model2_predictions.shape}")
    
    # ===== MODEL 3: Classify Functions =====
    print("\n[MODEL 3] Running multilabel classification...")
    model3_result = predict_function_from_model2(model2_predictions, threshold=threshold)
    
    if model3_result["model_loaded"]:
        print(f"✓ Predicted labels: {', '.join(model3_result['labels']) if model3_result['labels'] else 'None above threshold'}")
    else:
        print("⚠️ Model 3 not loaded - using placeholder")
    
    print(f"\n{'='*60}")
    print("✅ Pipeline Complete!")
    print(f"{'='*60}\n")
    
    return {
        "model1": model1_result,
        "model2": model2_result,
        "model3": model3_result,
        "sequence_id": seq_id,
        "sequence_length": len(sequence)
    }
# ==================== API Models ====================
class ProteinInput(BaseModel):
    sequence: str
    seq_id: str = None
class PipelineResponse(BaseModel):
    seq_id: str
    sequence: str
    model1_output: dict
    model2_output: dict
    model3_output: dict
    pipeline_status: str
# ==================== API Endpoints ====================
@app.get("/")
def root():
    return {
        "message": "Protein Analysis Pipeline API",
        "endpoints": {
            "/analyze": "POST - Run full 3-model pipeline",
            "/health": "GET - Check API health"
        }
    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "cached_embeddings": len(embedding_dict),
        "model2_loaded": model2 is not None,
        "model3_loaded": model3_classifier is not None
    }
@app.post("/api/analyze", response_model=PipelineResponse)
async def analyze_protein(protein_input: ProteinInput):
    """Run complete 3-model pipeline"""
    try:
        sequence = protein_input.sequence.upper().strip()
        
        # Auto-generate seq_id if not provided (using hash of sequence)
        if protein_input.seq_id:
            seq_id = protein_input.seq_id
        else:
            import hashlib
            seq_id = f"seq_{hashlib.md5(sequence.encode()).hexdigest()[:8]}"
            print(f"ℹ️ Auto-generated seq_id: {seq_id}")
        
        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_amino_acids for aa in sequence):
            raise HTTPException(
                status_code=400,
                detail="Invalid sequence. Use only valid amino acid letters."
            )
        
        print(f"\n{'='*50}")
        print(f"Processing sequence: {seq_id}")
        print(f"{'='*50}")
        
        # ===== MODEL 1: Generate Embedding =====
        print("\n[MODEL 1] Generating ESM2 embedding...")
        embedding = get_embedding(sequence, seq_id)
        
        model1_output = {
            "embedding": embedding.tolist(),  # Convert tensor to list for JSON
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
            "device": "cpu",
            "first_10_values": embedding[:10].tolist()
        }
        print(f"✓ Model 1 complete - Embedding shape: {embedding.shape}")
        
        # ===== MODEL 2: Neural Network Prediction =====
        print("\n[MODEL 2] Running neural network...")
        model2_result = predict_structure_from_embedding(embedding)
        
        if not model2_result["model_loaded"]:
            raise HTTPException(
                status_code=500,
                detail="Model 2 not loaded. Please ensure model2.pt exists."
            )
        
        model2_output = {
            "predictions": model2_result["raw_output"],  # 26121 values
            "shape": list(model2_result["predictions"].shape),
            "model_status": "loaded",
            "output_summary": {
                "min": float(np.min(model2_result["predictions"])),
                "max": float(np.max(model2_result["predictions"])),
                "mean": float(np.mean(model2_result["predictions"])),
                "positive_count": int(np.sum(model2_result["predictions"] > 0.5))
            }
        }
        
        # Optional: Annotate top GO terms
        if len(model2_result["predictions"]) == len(GO_INDEX_MAPPING):
            top_indices = np.argsort(model2_result["predictions"])[-5:][::-1]  # top 5
            top_go_terms = [
                {
                "index": int(idx),
                "go_id": GO_INDEX_MAPPING[idx]["go_id"],
                "go_name": GO_INDEX_MAPPING[idx]["go_name"],
                "score": float(model2_result["predictions"][idx]),
                }
                for idx in top_indices
                ]
            model2_output["top_go_terms"] = top_go_terms

        
        print(f"✓ Model 2 complete - Output shape: {model2_result['predictions'].shape}")
        
        # ===== MODEL 3: Multilabel Classification =====
        print("\n[MODEL 3] Running multilabel classification...")
        model3_result = predict_function_from_model2(model2_result["predictions"], threshold=0.5)
        
        model3_output = {
            "labels": model3_result["labels"],
            "probabilities": model3_result["probabilities"],
            "binary_predictions": model3_result["binary_predictions"],
            "model_loaded": model3_result["model_loaded"],
            "threshold": model3_result.get("threshold", 0.5)
        }
        
        if model3_result["model_loaded"]:
            print(f"✓ Model 3 complete - Labels: {', '.join(model3_result['labels']) if model3_result['labels'] else 'None'}")
        else:
            print(f"✓ Model 3 complete (placeholder)")
        
        print(f"\n{'='*50}")
        print("Pipeline completed successfully!")
        print(f"{'='*50}\n")
        
        return PipelineResponse(
            seq_id=seq_id,
            sequence=sequence,
            model1_output=model1_output,
            model2_output=model2_output,
            model3_output=model3_output,
            pipeline_status="success"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# ==================== Run Server ====================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Protein Analysis Pipeline API")
    print("="*60)
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🏠 Frontend: Open index.html in your browser")
    print("🔗 API Base URL: http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)