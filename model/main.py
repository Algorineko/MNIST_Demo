import torch
from PIL import Image
from torchvision import transforms
from multi_nn import Net
# 加载模型
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 加载并处理图像
image = Image.open('test_num.png').convert('L')
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
image = transform(image)
image = image.unsqueeze(0)  # 因为模型期望的输入是批量的图像，所以我们需要添加一个额外的维度

# 预测
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print(f'数字识别为: {predicted.item()}')