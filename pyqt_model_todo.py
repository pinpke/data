import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import models, transforms
"""
本模块实现了一个基于 PyQt5 的简单图像查看器应用。

类说明:
    ImageViewer(QWidget): QWidget 的子类，允许用户打开并显示电脑中的图像，并选择模型进行预测。

用法:
    直接运行此脚本以启动图像查看器窗口。点击“打开图片”按钮选择并显示一张图片，选择模型后点击“预测”按钮进行图片类型预测。

主要组件:
    - QLabel: 用于显示选中的图片或提示信息。
    - QPushButton: 用于触发文件对话框以打开图片。
    - QVBoxLayout: 垂直排列标签和按钮。
    - QComboBox: 用于选择模型（如 lenet、vgg、resnet 等）。
    - QPushButton: “预测”按钮，点击后对当前图片进行类型预测。
方法说明:
    ImageViewer.open_image():
        打开文件对话框让用户选择图片文件，并在标签中显示所选图片。
    ImageViewer.predict_image():
        根据选择的模型对当前图片进行类型预测，并弹窗显示预测结果。
"""

# 假设有三个模型的简单实现（实际使用时请替换为真实模型）
class ResNetModel:
    def __init__(self, weight_path):
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model.eval()
    def predict(self, img):
        with torch.no_grad():
            output = self.model(img)
            prob = torch.softmax(output, 1)
            pred = torch.argmax(prob, 1).item()
            confidence = prob[0, pred].item()
        return pred, confidence

class RealResNetModel:
    def __init__(self, weight_path):
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model.eval()
    def predict(self, img):
        with torch.no_grad():
            output = self.model(img)
            prob = torch.softmax(output, 1)
            pred = torch.argmax(prob, 1).item()
            confidence = prob[0, pred].item()
        return pred, confidence

from transformers import ViTForImageClassification, ViTImageProcessor

class RealViTModel:
    def __init__(self, weight_path):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=2
        )
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model.eval()

    def predict(self, img):
        # img为PIL.Image对象
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred].item()
        return pred, confidence

# 模型字典
MODEL_DICT = {
    "scresnet": ResNetModel("best_scresnet50_smoke.pth"),
    "vit": RealViTModel("best_vit_smoke.pth"),
    "resnet": RealResNetModel("best_resnet50_smoke.pth"),
}

def preprocess_image(image_path, model_name):
    img = Image.open(image_path).convert("RGB")
    if model_name == "vit":
        return img
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        return img

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QComboBox, QMessageBox
import os
import torch
from torchvision import transforms
from PIL import Image
import random
from torchvision import models
from PyQt5.QtWidgets import QHBoxLayout
class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt 图像查看器")
        self.resize(800, 400)

        # 图片显示区
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(350, 350)

        # 默认显示 test.jpg
        default_image = "./test.jpg"
        if os.path.exists(default_image):
            pixmap = QPixmap(default_image)
            self.label.setPixmap(pixmap)
            self.current_image_path = default_image
        else:
            self.label.setText("请打开一张图片")
            self.current_image_path = None

        # 结果显示区
        self.result_label = QLabel("预测结果：", self)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumWidth(200)

        # 模型选择框
        self.model_combo = QComboBox(self)
        self.model_combo.addItems(MODEL_DICT.keys())

        # 打开图片按钮
        self.open_button = QPushButton("打开图片", self)
        self.open_button.clicked.connect(self.open_image)

        # 预测按钮
        self.predict_button = QPushButton("预测", self)
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(self.current_image_path is not None)

        # 控件区布局（竖直排列：打开图片、模型选择、预测按钮、结果）
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(self.predict_button)
        control_layout.addWidget(self.result_label)
        control_layout.addStretch()

        # 主布局：水平分为图片区和右侧区
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.label, stretch=1)
        main_layout.addLayout(control_layout, stretch=1)
        self.setLayout(main_layout)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.label.setPixmap(pixmap)
            self.current_image_path = file_name
            self.predict_button.setEnabled(True)
            self.result_label.setText("预测结果：")
        else:
            if not self.current_image_path:
                self.label.setText("未选择图片")
                self.predict_button.setEnabled(False)
            self.result_label.setText("预测结果：")

    def predict_image(self):
        if not self.current_image_path:
            self.result_label.setText("请先选择一张图片！")
            return
        model_name = self.model_combo.currentText()
        model = MODEL_DICT.get(model_name)
        img = preprocess_image(self.current_image_path, model_name)
        pred, prob = model.predict(img)
        classes = ['非吸烟', '吸烟']
        result = classes[pred]
        self.result_label.setText(f"预测结果：{result}，概率：{prob:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())