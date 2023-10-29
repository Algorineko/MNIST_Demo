from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
from torchvision import transforms
from multi_nn import Net
import io
import torch.nn.functional as F
import uvicorn

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",  # Vue应用运行的地址
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 加载模型
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

@app.post("/recognition")
async def digital_recognition(image: UploadFile = File(...)):
    print(f"Received image: {image.filename}")
    # 读取图片数据
    image_data = await image.read()
    # 将图片数据转为PIL对象
    image_pil = Image.open(io.BytesIO(image_data)).convert('L')
    # 对图片进行预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image_pil)
    image_tensor = image_tensor.unsqueeze(0)  # 因为模型期望的输入是批量的图像，所以我们需要添加一个额外的维度

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        # 获取每个类别的概率
        probabilities = F.softmax(output.data, dim=1)
        probabilities = probabilities.numpy().tolist()  # 转换为列表
        # 将概率转换为百分比
        probabilities_percent = [format(prob * 100, '.4f') for prob in probabilities[0]]

    # 返回识别结果和每个类别的概率百分比
    return {"message": predicted.item(), "probabilities": probabilities_percent}

# 启动进入model文件夹