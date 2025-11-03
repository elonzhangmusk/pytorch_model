import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from models.googlenet import GoogleNet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# 1. 参数配置
# -----------------------------
MODEL_PATH = "/home/zhanghangning/pytorch/checkpoint/googlenet/Monday_27_October_2025_11h_56m_21s/googlenet-100-regular.pth"
IMAGE_DIR = "/home/zhanghangning/pytorch/left-eye/test/healthy"
SAVE_DIR = "/home/zhanghangning/pytorch/gradcam_results"
NUM_CLASSES = 2
USE_CUDA = True
FLIP_CAM = True  # True 可以反转热力图颜色

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 2. 加载模型
# -----------------------------
device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
model = GoogleNet(num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -----------------------------
# 3. Grad-CAM设置
# -----------------------------
target_layer = model.inception5b
cam = GradCAM(model=model, target_layers=[target_layer])  # 不要 use_cuda
# -----------------------------
# 4. 图片预处理函数
# -----------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    img_np = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    return tensor, img_np

# -----------------------------
# 5. 批量生成Grad-CAM可视化
# -----------------------------
image_paths = [os.path.join(IMAGE_DIR, f)
               for f in os.listdir(IMAGE_DIR)
               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for path in image_paths:
    img_tensor, img_np = preprocess_image(path)

    # 预测类别
    with torch.no_grad():
        output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Grad-CAM
    grayscale_cam = cam(input_tensor=img_tensor,
                        targets=[ClassifierOutputTarget(pred_class)])
    mask = grayscale_cam[0, :, :]  # 取出单张图



    # 可视化
    visualization = show_cam_on_image(img_np, mask, use_rgb=True)

    # 保存
    save_path = os.path.join(SAVE_DIR, os.path.basename(path))
    plt.imsave(save_path, visualization)

print(f"✅ Grad-CAM 可视化结果已保存到: {SAVE_DIR}")
