import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import time

# # 加载预训练的ResNet50模型
# def load_model():
#     # 创建ResNet50模型，指定输出为2类
#     model = models.resnet50(weights=None)  # 不加载ImageNet权重
#     model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层为二分类
#     model.load_state_dict(torch.load('best_resnet50_smoke.pth'))  # 加载训练好的权重
#     model.eval()  # 设置为评估模式
#     return model
# 加载ViT模型和图像处理器
def load_model():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=2
    )
    model.load_state_dict(torch.load('best_vit_smoke.pth', map_location='cpu'))  # 你的ViT权重
    model.eval()
    return model, processor
# 定义图像预处理流程，需与训练时一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载OpenCV自带的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def main():
#     # 加载模型并设置设备
#     model = load_model()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
    
#     # 打开摄像头
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # 转为灰度图用于人脸检测
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.2, 4)  # 检测人脸
        
#         for (x, y, w, h) in faces:
#             # 绘制人脸框
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
#             # 提取人脸区域
#             face_roi = frame[y:y+h, x:x+w]
            
#             # 转为PIL图像
#             face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
#             # 预处理
#             face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
#             # 预测是否吸烟
#             with torch.no_grad():
#                 outputs = model(face_tensor)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 pred_prob, pred_class = torch.max(probabilities, 1)
                
#                 # 获取预测结果
#                 is_smoking = pred_class.item() == 1  # 1为吸烟，0为非吸烟
#                 confidence = pred_prob.item() * 100  # 置信度百分比
                
#                 # 在图像上显示结果
#                 status = "Smoking" if is_smoking else "Non-smoking"
#                 color = (0, 0, 255) if is_smoking else (0, 255, 0)
#                 cv2.putText(frame, f"{status}: {confidence:.2f}%", 
#                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
#         # 显示结果窗口
#         cv2.imshow('Smoke Detection', frame)
        
#         # 按'q'退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # 释放摄像头和窗口资源
#     cap.release()
#     cv2.destroyAllWindows()
def main():
    # 加载模型和处理器
    model, processor = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

            # 记录推理开始时间
            start_time = time.time()

            # ViT预处理
            inputs = processor(images=face_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prob = torch.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1).item()
                confidence = prob[0, pred].item() * 100

            # 记录推理结束时间
            end_time = time.time()
            infer_time = (end_time - start_time) * 1000  # 毫秒

            is_smoking = pred == 1  # 1为吸烟，0为非吸烟
            status = "Smoking" if is_smoking else "Non-smoking"
            color = (0, 0, 255) if is_smoking else (0, 255, 0)
            cv2.putText(frame, f"{status}: {confidence:.2f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # 显示推理时间
            cv2.putText(frame, f"Infer: {infer_time:.1f} ms",
                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Smoke Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()