import torch
import torch.nn as nn

# Load your model
state_dict = torch.load("./models/Model_1.pt", map_location="cpu")

# If it's just the state_dict, rebuild a dummy architecture first (example)
# ⚠️ Replace this with your actual model class if you have it in your project
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

# Initialize model and load weights
model = Model2Network(640,26121)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ✅ Print architecture
print("\n=== Model Architecture ===")
print(model)

# ✅ Print layer-wise names and shapes
print("\n=== Layer Details ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# ✅ Run a test inference
print("\n=== Test Inference ===")
dummy_input = torch.randn(1, 640)  # match input size
output = model(dummy_input)
print("Output shape:", output.shape)
print("Output values:", output)
