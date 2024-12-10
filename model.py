import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from torchvision import models

class ImageClassifier:
    def __init__(self):
        self.model =models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img = Image.open(image).convert('RGB')
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            out = self.model(batch_t)
        
        _, index = torch.max(out, 1)
        return index.item()