import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Chuyển đổi và chuẩn hóa ảnh MNIST
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize ảnh lên 32x32
    transforms.ToTensor(),        # Chuyển đổi ảnh thành tensor
    # Chuẩn hóa ảnh với giá trị trung bình 0.5 và độ lệch chuẩn 0.5
    transforms.Normalize((0.5,), (0.5,))
])


# Tải tập dữ liệu MNIST
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Định nghĩa mô hình LeNet-5


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Đầu vào (1, 32, 32), đầu ra (6, 32, 32)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # Đầu ra (6, 16, 16)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Đầu vào (6, 16, 16), đầu ra (16, 12, 12)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Đầu vào (16, 8, 8), đầu ra (120, 4, 4)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(120 * 4 * 4, 84)
        # Fully Connected Layer 2 (đầu ra 10 lớp)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # Lớp tích chập 1 + kích hoạt tanh
        x = self.pool(x)               # Lớp gộp 1 (average pooling)
        x = torch.tanh(self.conv2(x))  # Lớp tích chập 2 + kích hoạt tanh
        x = self.pool(x)               # Lớp gộp 2 (average pooling)
        x = torch.tanh(self.conv3(x))  # Lớp tích chập 3 + kích hoạt tanh
        x = x.view(-1, 120 * 4 * 4)    # Flatten đầu vào
        # Fully Connected Layer 1 + kích hoạt tanh
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)                # Fully Connected Layer 2 (đầu ra)
        return x


# Tạo mô hình, hàm mất mát và optimizer
model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader, 0):
        # Chuyển ảnh và nhãn vào thiết bị
        optimizer.zero_grad()

        # Tiến hành forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Tiến hành backward và tối ưu
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # In ra mỗi 100 batch
            print(
                f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Hoàn thành huấn luyện!')

# Đánh giá mô hình trên tập kiểm tra
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Độ chính xác trên tập kiểm tra: {100 * correct / total:.2f}%')
