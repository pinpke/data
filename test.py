import torch
from torchvision import transforms, models
from PIL import Image

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型（以ResNet50为例，类别数2）
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_resnet50_smoke.pth', map_location=device))
model = model.to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图片
img = Image.open('test.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # 增加batch维度

# 推理
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, 1).item()
    prob = torch.softmax(output, 1)[0, pred].item()

# 类别映射（根据你的训练集实际类别顺序）
idx_to_class = {0: 'non-smoking', 1: 'smoking'}
print(f'预测结果: {idx_to_class[pred]}, 置信度: {prob*100:.2f}%')