import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from scresnet50 import ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = datasets.ImageFolder(root='output/train', transform=transform)
print("类别到标签的映射：", dataset.class_to_idx)

# 划分训练集和验证集
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载测试集
test_dataset = datasets.ImageFolder(root='output/test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载ResNet50模型并修改输出层
# model = models.resnet50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(model.fc.in_features, 2)
model = ResNet50(num_classes=2) #scresnet50.py
model = model.to(device)

#optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', ncols=80)
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loss = running_loss / (train_pbar.n + 1)
        train_acc = 100 * correct / total
        train_pbar.set_postfix({'Train Loss': train_loss, 'Train Acc': f'{train_acc:.2f}%'})
    print(f'Epoch {epoch + 1}/{num_epochs}, Final Train Loss: {train_loss:.4f}, Final Train Acc: {train_acc:.2f}%')
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', ncols=80)
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_dataloader)
    val_acc = 100 * val_correct / val_total
    print(f'[Val] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_scresnet50_smoke.pth')

# 绘制准确率和损失曲线
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss_list, label='Train Loss')
plt.plot(epochs, val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1,2,2)
plt.plot(epochs, train_acc_list, label='Train Acc')
plt.plot(epochs, val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()

# 输出最高准确率
print(f'验证集最高准确率: {max(val_acc_list):.2f}%')

# 测试集评估
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
test_loss /= len(test_dataloader)
test_acc = 100 * test_correct / test_total
print(f'[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

# 保存模型参数
torch.save(model.state_dict(), 'scresnet50_smoke.pth')
print('模型已保存为 resnet50_smoke.pth')