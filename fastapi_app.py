from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a pre-trained model (replace with your own model if you have one)
try:
    model = models.resnet18(pretrained=True)
    model.eval()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image details
        logger.info(f"Received image: format={image.format}, size={image.size}, mode={image.mode}")
        
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get the class name (replace with your own class names if you have them)
        class_names = ['class1', 'class2', 'class3']  # Example class names
        try:
            predicted_class = class_names[predicted.item()]
        except IndexError:
            logger.error(f"Predicted index {predicted.item()} out of range for class_names")
            raise HTTPException(status_code=500, detail="Prediction error: Invalid class index")

        return JSONResponse(content={"class": predicted_class, "confidence": outputs[0][predicted].item()})
    except Exception as e:
        logger.exception("An error occurred during prediction")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)