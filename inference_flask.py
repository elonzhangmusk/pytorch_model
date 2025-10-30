from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from models.googlenet import GoogleNet  # 导入你提供的 GoogleNet 类

app = Flask(__name__)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
net = GoogleNet(num_classes=2)  # 二分类
weights_path = '/home/zhanghangning/pytorch/preTrainedCheckpoints/googlenet-1378be20.pth'

# 加载权重
state_dict = torch.load(weights_path, map_location=device)
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
net.to(device)
net.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5813, 0.4787, 0.4483],
                         std=[0.1629, 0.1960, 0.2080])
])

# 类别名称
class_names = ['class0', 'class1']  # 根据你的数据修改

@app.route('/')
def index():
    return render_template('flask.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    img_file = request.files['image']
    img = Image.open(img_file).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(img)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_class = class_names[pred_idx]
    
    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
