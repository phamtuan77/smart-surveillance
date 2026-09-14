import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchvision.models as models

class FeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        # Dùng ResNet18 pretrained làm backbone trích đặc trưng (baseline đơn giản)
        #self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        self.model.fc = torch.nn.Identity()  # bỏ lớp phân loại, chỉ lấy vector đặc trưng
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        image: ảnh crop của 1 người, dạng numpy array (OpenCV, BGR)
        return: vector đặc trưng (numpy array, 512 chiều)
        """
        img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(tensor)

        return feature.squeeze(0).cpu().numpy()