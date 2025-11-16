# main.py
import os
import io
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import uvicorn

# ------------------------------
# API key (optional security)
# ------------------------------
API_KEY = "mysecret123"  # Change this to your secret key

# ------------------------------
# Initialize FastAPI
# ------------------------------
app = FastAPI(title="Image Classification API")

# ------------------------------
# Load model
# ------------------------------
checkpoint = torch.load("best_model.pth", map_location="cpu")
classes = checkpoint["classes"]

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ------------------------------
# Image preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI Model API is running ✔️"}

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(...)):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    img = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    return {"prediction": classes[pred.item()]}

# ------------------------------
# Run server (Render compatible)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
