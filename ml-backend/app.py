from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import esm
import pickle
import os
import numpy as np
from typing import List
import json
app = FastAPI(title="Protein Function Prediction API")
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# LOAD ALL THREE COMPONENTS
# =====================================================
class ModelManager:
    def __init__(self):
        print("=" * 60)
        print("🔄 LOADING MODELS...")
        print("=" * 60)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📍 Using device: {self.device}")
        
        # ========================================
        # ESM-2 MODEL (For Embeddings)
        # ========================================
        print("\n🧬 Loading ESM2-150M model...")
        self.esm_model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        print("✅ ESM-2 model loaded successfully!")
        
        # Load/initialize embedding cache
        self.emb_path = "embeddings_150M.pkl"
        if os.path.exists(self.emb_path):
            with open(self.emb_path, "rb") as f:
                self.embedding_dict = pickle.load(f)
            print(f"✅ Loaded {len(self.embedding_dict)} cached embeddings")
        else:
            self.embedding_dict = {}
            print("📝 No existing embeddings found — starting fresh")
        
        # ========================================
        # MODEL 1: Embeddings → Intermediate Output
        # ========================================
        print("\n🎯 Loading Model 1...")
        
        # TODO: Replace this with your Model 1 loading code
        # Example:
        # self.model_1 = torch.load('models/model_1.pth', map_location=self.device)
        # self.model_1.eval()
        
        self.model_1 = None  # REPLACE THIS with your actual model
        print("⚠️  Model 1 not loaded yet - add your loading code!")
        
        # ========================================
        # MODEL 2: Intermediate → GO Predictions
        # ========================================
        print("\n🎯 Loading Model 2...")
        
        # TODO: Replace this with your Model 2 loading code
        # Example:
        # self.model_2 = torch.load('models/model_2.pth', map_location=self.device)
        # self.model_2.eval()
        
        self.model_2 = None  # REPLACE THIS with your actual model
        print("⚠️  Model 2 not loaded yet - add your loading code!")
        
        # ========================================
        # LOAD GO TERM MAPPING
        # ========================================
        print("\n📚 Loading GO term mapping...")
        
        # TODO: Replace with your actual GO mapping file
        # with open('go_terms_mapping.json', 'r') as f:
        #     self.go_mapping = json.load(f)
        
        # Placeholder GO mapping - REPLACE THIS
        self.go_mapping = {
            "GO:0009274": {"name": "Peptidoglycan-based cell wall", "ontology": "CC"},
            "GO:0071944": {"name": "Cell periphery", "ontology": "CC"},
            "GO:0031625": {"name": "Ubiquitin protein ligase binding", "ontology": "MF"},
            "GO:0006281": {"name": "DNA repair", "ontology": "BP"},
            "GO:0005524": {"name": "ATP binding", "ontology": "MF"},
            "GO:0016020": {"name": "Membrane", "ontology": "CC"},
            "GO:0003677": {"name": "DNA binding", "ontology": "MF"},
            "GO:0046872": {"name": "Metal ion binding", "ontology": "MF"},
        }
        
        # List of GO terms in same order as model output
        self.go_terms_list = list(self.go_mapping.keys())
        
        print(f"✅ Loaded {len(self.go_terms_list)} GO terms")
        print("=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
    
    def get_embedding(self, sequence: str, seq_id: str):
        """
        Generate or retrieve cached ESM-2 embedding for a sequence
        Returns: PyTorch tensor of shape (640,) for ESM2-150M
        """
        # Check cache first
        if seq_id in self.embedding_dict:
            print(f"✓ Found cached embedding for {seq_id}")
            return self.embedding_dict[seq_id]
        
        # Generate new embedding
        print(f"⚡ Generating new embedding for {seq_id}")
        data = [(seq_id, sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[30])
            token_representations = results["representations"][30]
        
        # Mean pooling over sequence length
        embedding = token_representations.mean(1).squeeze().cpu()
        
        # Cache the embedding
        self.embedding_dict[seq_id] = embedding
        with open(self.emb_path, "wb") as f:
            pickle.dump(self.embedding_dict, f)
        
        print(f"✓ Generated and cached embedding for {seq_id}")
        return embedding
# Initialize models once at startup
print("\n🚀 Starting Protein Function Prediction API...\n")
models = ModelManager()
# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================
class PredictionRequest(BaseModel):
    sequence: str
class GOPrediction(BaseModel):
    GO_term: str
    name: str
    probability: float
    ontology: str
class PredictionResponse(BaseModel):
    success: bool
    sequence_length: int
    predictions: List[GOPrediction]
    threshold_used: float
    total_above_threshold: int
# =====================================================
# MAIN PREDICTION ENDPOINT
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict_protein_function(request: PredictionRequest):
    """
    Three-step prediction pipeline:
    1. Sequence → ESM-2 Embeddings (640-dim)
    2. Embeddings → Model 1 → Intermediate Output
    3. Intermediate → Model 2 → GO Predictions
    4. Filter: Keep only if probability > 0.7
    """
    try:
        sequence = request.sequence.strip().upper()
        
        # Validate sequence
        if len(sequence) < 10:
            raise HTTPException(status_code=400, detail="Sequence too short (minimum 10 amino acids)")
        
        # Check for valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aa for aa in sequence):
            raise HTTPException(status_code=400, detail="Invalid amino acids in sequence")
        
        print(f"\n{'='*60}")
        print(f"🧬 Processing sequence of length {len(sequence)}")
        print(f"{'='*60}")
        
        # STEP 1: Get ESM-2 embeddings (with caching)
        print("⚡ Step 1: Converting sequence to ESM-2 embeddings...")
        seq_id = f"seq_{hash(sequence) % 1000000}"  # Generate unique ID
        embeddings = models.get_embedding(sequence, seq_id)
        print(f"✅ Embeddings generated: shape {embeddings.shape}")
        
        # STEP 2: Pass embeddings through Model 1
        print("⚡ Step 2: Running Model 1...")
        model1_output = run_model_1(embeddings)
        print(f"✅ Model 1 output generated")
        
        # STEP 3: Pass Model 1 output through Model 2
        print("⚡ Step 3: Running Model 2 to get GO predictions...")
        go_probabilities = run_model_2(model1_output)
        print(f"✅ GO predictions generated: {len(go_probabilities)} terms")
        
        # STEP 4: Filter predictions with threshold = 0.7
        threshold = 0.7
        print(f"⚡ Step 4: Filtering predictions (threshold = {threshold})...")
        filtered_predictions = filter_predictions_by_threshold(go_probabilities, threshold)
        print(f"✅ Found {len(filtered_predictions)} predictions above threshold")
        print(f"{'='*60}\n")
        
        return PredictionResponse(
            success=True,
            sequence_length=len(sequence),
            predictions=filtered_predictions,
            threshold_used=threshold,
            total_above_threshold=len(filtered_predictions)
        )
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# =====================================================
# STEP 2: MODEL 1 INFERENCE
# =====================================================
def run_model_1(embeddings):
    """
    Take ESM-2 embeddings (640-dim) and pass through Model 1
    
    TODO: Replace this function with your actual Model 1 inference
    
    Input: PyTorch tensor of shape (640,)
    Output: Whatever your Model 1 outputs (e.g., features, intermediate representation)
    """
    
    # ========================================
    # YOUR MODEL 1 CODE GOES HERE
    # ========================================
    
    # Example implementation:
    # with torch.no_grad():
    #     # Ensure embeddings are on correct device
    #     embeddings = embeddings.to(models.device)
    #     
    #     # Add batch dimension if needed
    #     if embeddings.dim() == 1:
    #         embeddings = embeddings.unsqueeze(0)  # Shape: (1, 640)
    #     
    #     # Run Model 1
    #     model1_output = models.model_1(embeddings)
    #     
    #     # Remove batch dimension if needed
    #     if model1_output.dim() == 2 and model1_output.shape[0] == 1:
    #         model1_output = model1_output.squeeze(0)
    # 
    # return model1_output
    
    # ========================================
    # PLACEHOLDER: Mock output for testing
    # ========================================
    # Replace this with your actual Model 1 inference code
    
    # Assuming Model 1 outputs some feature vector
    mock_output = torch.randn(512).to(models.device)  # Example: 512-dim output
    
    return mock_output
# =====================================================
# STEP 3: MODEL 2 INFERENCE
# =====================================================
def run_model_2(model1_output):
    """
    Take Model 1 output and pass through Model 2 to get GO predictions
    
    TODO: Replace this function with your actual Model 2 inference
    
    Input: Model 1's output (whatever shape it produces)
    Output: Array of probabilities for each GO term
    """
    
    # ========================================
    # YOUR MODEL 2 CODE GOES HERE
    # ========================================
    
    # Example implementation:
    # with torch.no_grad():
    #     # Ensure input is on correct device
    #     model1_output = model1_output.to(models.device)
    #     
    #     # Add batch dimension if needed
    #     if model1_output.dim() == 1:
    #         model1_output = model1_output.unsqueeze(0)
    #     
    #     # Run Model 2
    #     logits = models.model_2(model1_output)
    #     
    #     # Convert to probabilities
    #     probabilities = torch.sigmoid(logits)  # For multi-label classification
    #     # OR probabilities = torch.softmax(logits, dim=-1)  # For multi-class
    #     
    #     # Convert to numpy array
    #     probabilities = probabilities.cpu().numpy().flatten()
    # 
    # return probabilities
    
    # ========================================
    # PLACEHOLDER: Mock probabilities for testing
    # ========================================
    # Replace this with your actual Model 2 inference code
    
    num_go_terms = len(models.go_terms_list)
    mock_probabilities = np.random.uniform(0.3, 0.95, num_go_terms)
    
    return mock_probabilities
# =====================================================
# STEP 4: FILTER BY THRESHOLD (SIMPLE IF-ELSE)
# =====================================================
def filter_predictions_by_threshold(probabilities, threshold=0.7):
    """
    Filter GO predictions using simple if-else logic
    
    Rule: if probability > 0.7: keep prediction
          else: skip prediction
    
    Input: Array of probabilities (one per GO term)
    Output: List of GOPrediction objects with prob > threshold
    """
    filtered_predictions = []
    
    for idx, probability in enumerate(probabilities):
        # Simple if-else check as requested
        if probability > threshold:
            # Get GO term information
            go_term = models.go_terms_list[idx]
            go_info = models.go_mapping[go_term]
            
            # Create prediction object
            prediction = GOPrediction(
                GO_term=go_term,
                name=go_info["name"],
                probability=float(probability),
                ontology=go_info["ontology"]
            )
            
            filtered_predictions.append(prediction)
            
            print(f"  ✓ {go_term}: {probability:.3f} (above threshold)")
        else:
            # Skip predictions below threshold
            continue
    
    # Sort by probability (highest first)
    filtered_predictions.sort(key=lambda x: x.probability, reverse=True)
    
    return filtered_predictions
# =====================================================
# HEALTH CHECK ENDPOINTS
# =====================================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "esm2": True,
            "model_1": models.model_1 is not None,
            "model_2": models.model_2 is not None
        },
        "device": str(models.device),
        "num_go_terms": len(models.go_terms_list),
        "cached_embeddings": len(models.embedding_dict),
        "threshold": 0.7
    }
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Protein Function Prediction API",
        "version": "3.0.0",
        "pipeline": [
            "1. Sequence → ESM-2 Embeddings (640-dim, cached)",
            "2. Embeddings → Model 1 → Intermediate Output",
            "3. Intermediate → Model 2 → GO Predictions",
            "4. Filter by threshold (>0.7)"
        ],
        "endpoints": {
            "POST /predict": "Predict GO terms for a protein sequence",
            "GET /health": "Check API health",
            "GET /": "API information"
        }
    }
# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Protein Function Prediction API is ready!")
    print("📍 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)