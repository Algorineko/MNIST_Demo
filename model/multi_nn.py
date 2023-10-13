import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义一个简单的卷积神经网络模型
class Net(nn.Module): # 继承自Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义数据转换和加载数据
transform = transforms.Compose([
    transforms.ToTensor(), # 转化为tensor
    transforms.Normalize((0.5,), (0.5,)) # 归一化
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型和优化器
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# tensor=torch.randn([64,1,28,28])
# output=model(tensor)
# print(output.shape)

# 定义一个函数来训练模型
def train_net():
    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad() # 梯度归零
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # 反向传播
            optimizer.step() # 更新
            # backward在step之前即可
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试集准确率: {100 * correct / total}%')
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model.pth')
if __name__ == '__main__':
    train_net()