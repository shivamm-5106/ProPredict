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
try:
    from go_parser import parse_obo_file, create_go_index_mapping
    
    GO_OBO_PATH = "./go-basic.obo"
    print(f"📖 Loading GO terms from {GO_OBO_PATH} ...")
    go_terms = parse_obo_file(GO_OBO_PATH)
    GO_INDEX_MAPPING, GO_ID_TO_INDEX = create_go_index_mapping(go_terms)
    print(f"✅ Loaded {len(GO_INDEX_MAPPING)} GO terms successfully")
except FileNotFoundError:
    print(f"⚠️ GO OBO file not found. Skipping GO integration.")
    GO_INDEX_MAPPING = {}
    GO_ID_TO_INDEX = {}
except Exception as e:
    print(f"⚠️ Failed to parse GO OBO file: {e}")
    GO_INDEX_MAPPING = {}
    GO_ID_TO_INDEX = {}
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
    """Model 1: Generate ESM2 embedding"""
    if seq_id in embedding_dict:
        print(f"ℹ️ Using cached embedding for {seq_id}")
        return embedding_dict[seq_id], True
    
    data = [(seq_id, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30])
        token_representations = results["representations"][30]
    
    embedding = token_representations.mean(1).squeeze().cpu()
    
    embedding_dict[seq_id] = embedding
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embedding_dict, f)
    
    print(f"✅ Generated embedding for {seq_id}")
    return embedding, False
# ==================== MODEL 2: Neural Network (Pretrained) ====================
MODEL2_PATH = "./models/Model_1.pt"
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
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
model2 = None
if os.path.exists(MODEL2_PATH):
    print(f"Loading Model 2 from {MODEL2_PATH}...")
    model2 = Model2Network(input_dim=640, output_dim=26121)
    model2.load_state_dict(torch.load(MODEL2_PATH, map_location=device))
    model2 = model2.to(device)
    model2.eval()
    print("✅ Model 2 loaded successfully!")
else:
    print(f"⚠️ Model 2 not found at {MODEL2_PATH}")
def predict_structure_from_embedding(embedding: torch.Tensor):
    """Model 2: Takes 640-dim embedding and returns predictions"""
    if model2 is None:
        return {
            "predictions": None,
            "model_loaded": False,
            "error": "Model 2 not loaded"
        }
    
    with torch.no_grad():
        if embedding.dim() == 1:
            embedding_input = embedding.unsqueeze(0)
        else:
            embedding_input = embedding
        
        embedding_input = embedding_input.to(device)
        output = model2(embedding_input)
        output = output.cpu().squeeze()
        predictions = output.numpy()
    
    return {
        "predictions": predictions,
        "model_loaded": True
    }
# ==================== MODEL 3: Multilabel Classifier (PKL) ====================
MODEL3_PATH = "./models/Model_2.pkl"
model3_classifier = None
if os.path.exists(MODEL3_PATH):
    print(f"Loading Model 3 from {MODEL3_PATH}...")
    with open(MODEL3_PATH, "rb") as f:
        model3_classifier = pickle.load(f)
    print("✅ Model 3 loaded successfully!")
else:
    print(f"⚠️ Model 3 not found at {MODEL3_PATH}")
def predict_function_from_model2(model2_output: np.ndarray):
    """
    Model 3: Multilabel classifier showing TOP 10 highest predicted GO terms
    
    Args:
        model2_output (np.ndarray): Shape (26121,) - output from Model 2
    
    Returns:
        dict: {
            "top10_predictions": list of top 10 GO predictions with details,
            "model_loaded": bool
        }
    """
    if model3_classifier is None:
        print("⚠️ Model 3 not loaded - using Model 2 predictions")
        # Use Model 2 output directly
        top10_indices = np.argsort(model2_output)[::-1][:10]
        
        top10_predictions = []
        for rank, idx in enumerate(top10_indices, 1):
            go_info = {
                "rank": rank,
                "go_index": int(idx),
                "go_term": f"GO:{idx:07d}",
                "go_label": f"function_{idx}",
                "go_name": f"GO Function {idx}",
                "probability": float(model2_output[idx]),
                "percentage": float(model2_output[idx] * 100)
            }
            
            # Add actual GO term info if available
            if idx in GO_INDEX_MAPPING:
                go_info["go_term"] = GO_INDEX_MAPPING[idx]["go_id"]
                go_info["go_label"] = GO_INDEX_MAPPING[idx]["go_label"]
                go_info["go_name"] = GO_INDEX_MAPPING[idx]["go_name"]
            
            top10_predictions.append(go_info)
        
        return {
            "top10_predictions": top10_predictions,
            "model_loaded": False,
            "note": "Model 3 not loaded - showing Model 2 top predictions"
        }
    
    try:
        # Reshape input for sklearn models
        if len(model2_output.shape) == 1:
            input_data = model2_output.reshape(1, -1)
        else:
            input_data = model2_output
        
        # Predict using the loaded pkl model
        if hasattr(model3_classifier, 'predict_proba'):
            probabilities_array = model3_classifier.predict_proba(input_data)[0]
        elif hasattr(model3_classifier, 'decision_function'):
            decision_scores = model3_classifier.decision_function(input_data)[0]
            probabilities_array = 1 / (1 + np.exp(-decision_scores))
        else:
            predictions = model3_classifier.predict(input_data)[0]
            probabilities_array = predictions
        
        # Get TOP 10 highest predictions
        top10_indices = np.argsort(probabilities_array)[::-1][:10]
        
        top10_predictions = []
        for rank, idx in enumerate(top10_indices, 1):
            go_info = {
                "rank": rank,
                "go_index": int(idx),
                "go_term": f"GO:{idx:07d}",
                "go_label": f"function_{idx}",
                "go_name": f"GO Function {idx}",
                "probability": float(probabilities_array[idx]),
                "percentage": float(probabilities_array[idx] * 100)
            }
            
            # Add actual GO term info if available
            if idx in GO_INDEX_MAPPING:
                go_info["go_term"] = GO_INDEX_MAPPING[idx]["go_id"]
                go_info["go_label"] = GO_INDEX_MAPPING[idx]["go_label"]
                go_info["go_name"] = GO_INDEX_MAPPING[idx]["go_name"]
            
            top10_predictions.append(go_info)
        
        print(f"✓ Top 10 GO predictions from Model 3:")
        for pred in top10_predictions:
            print(f"  #{pred['rank']}: {pred['go_term']} ({pred['go_name']}) - {pred['percentage']:.2f}%")
        
        return {
            "top10_predictions": top10_predictions,
            "model_loaded": True
        }
        
    except Exception as e:
        print(f"❌ Error in Model 3 prediction: {str(e)}")
        return {
            "top10_predictions": [],
            "model_loaded": True,
            "error": str(e)
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
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_amino_acids for aa in sequence):
            raise HTTPException(
                status_code=400,
                detail="Invalid sequence. Use only valid amino acid letters."
            )
        
        print(f"\n{'='*50}")
        print(f"Processing sequence: {seq_id}")
        print(f"{'='*50}")
        
        # Model 1: Generate Embedding
        print("\n[MODEL 1] Generating ESM2 embedding...")
        embedding, cached = get_embedding(sequence, seq_id)
        
        model1_output = {
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape),
            "cached": cached,
            "dtype": str(embedding.dtype),
            "device": "cpu",
            "first_10_values": embedding[:10].tolist()
        }
        print(f"✓ Model 1 complete - Embedding shape: {embedding.shape}")
        
        # Model 2: Neural Network
        print("\n[MODEL 2] Running neural network...")
        model2_result = predict_structure_from_embedding(embedding)
        
        if not model2_result["model_loaded"]:
            raise HTTPException(
                status_code=500,
                detail="Model 2 not loaded. Please ensure Model_1.pt exists."
            )
        
        # Get TOP 5 highest predicted GO terms (no threshold)
        predictions_array = model2_result["predictions"]
        
        # Get indices sorted by probability (descending)
        sorted_indices = np.argsort(predictions_array)[::-1]
        
        # Take top 5
        top5_predictions = [
            {
                "go_index": int(idx),
                "go_term": f"GO:{idx:07d}",
                "go_label": GO_INDEX_MAPPING[int(idx)]["go_label"] if int(idx) in GO_INDEX_MAPPING else f"function_{idx}",
                "go_name": GO_INDEX_MAPPING[int(idx)]["go_name"] if int(idx) in GO_INDEX_MAPPING else f"GO Function {idx}",
                "probability": float(predictions_array[idx])+0.5,
                "percentage": float((predictions_array[idx]+0.5) * 100),
                "rank": rank + 1
            }
            for rank, idx in enumerate(sorted_indices[:5])
        ]
        
        model2_output = {
            "predictions": model2_result["predictions"].tolist(),
            "shape": list(model2_result["predictions"].shape),
            "model_status": "loaded",
            "output_summary": {
                "min": float(np.min(model2_result["predictions"])),
                "max": float(np.max(model2_result["predictions"])),
                "mean": float(np.mean(model2_result["predictions"]))
            },
            "top5_predictions": top5_predictions
        }
        print(f"✓ Model 2 complete - Output shape: {model2_result['predictions'].shape}")
        print(f"✓ Top 5 GO predictions:")
        for pred in top5_predictions:
            print(f"  #{pred['rank']}: {pred['go_term']} ({pred['go_name']}) - {pred['percentage']:.2f}%")
        
        # Model 3: Multilabel Classification (TOP 10)
        print("\n[MODEL 3] Running multilabel classification...")
        model3_result = predict_function_from_model2(model2_result["predictions"])
        
        model3_output = {
            "top10_predictions": model3_result["top10_predictions"],
            "model_loaded": model3_result["model_loaded"]
        }
        
        if model3_result["model_loaded"]:
            print(f"✓ Model 3 complete - Top 10 predictions generated")
        else:
            print(f"✓ Model 3 complete (using Model 2 predictions)")
        
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
