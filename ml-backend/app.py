# In ModelLoader.__init__()
self.model = torch.load('your_model.pth', map_location=self.device)
self.model.eval()

with open('go_terms_mapping.json', 'r') as f:
    self.go_mapping = json.load(f)
    
# Get embeddings (if needed)
embeddings = get_esm2_embeddings(sequence)

# Run your model
logits = models.model(embeddings)
probabilities = torch.sigmoid(logits).cpu().numpy()

# Convert to GO predictions
# (code provided in guide)

