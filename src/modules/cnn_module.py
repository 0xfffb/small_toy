from torch import nn


class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        # 1. 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 输入通道数：1（灰度图像）
            out_channels=32,    # 输出通道数：32（提取32种特征）
            kernel_size=3,      # 卷积核大小：3x3
            stride=1,           # 步长：1（每次移动1个像素）
            padding=1           # 填充：1（保持输入输出尺寸相同）
        )
        # 输入：[batch_size, 1, 28, 28]
        # 输出：[batch_size, 32, 28, 28]
        
        # 2. ReLU激活层
        self.relu = nn.ReLU()
        # ReLU函数：f(x) = max(0, x)
        # 作用：增加非线性，保持维度不变
        
        # 3. 最大池化层
        self.pool = nn.MaxPool2d(
            kernel_size=2    # 池化窗口大小：2x2
            # stride默认等于kernel_size
            # 在2x2的窗口中取最大值
        )
        # 输入：[batch_size, 32, 28, 28]
        # 输出：[batch_size, 32, 14, 14]
        
        # 4. 全连接层
        self.fc = nn.Linear(
            in_features=32 * 14 * 14,  # 输入特征数：6272
            out_features=10            # 输出特征数：10（类别数）
        )
        # 输入：[batch_size, 6272]
        # 输出：[batch_size, 10]

    def forward(self, x):
        # x的初始形状：[batch_size, 1, 28, 28]
        x = self.conv1(x)    # -> [batch_size, 32, 28, 28]
        x = self.relu(x)     # 形状保持不变
        x = self.pool(x)     # -> [batch_size, 32, 14, 14]
        # 展平操作：将多维特征图转换为一维向量
        x = x.view(-1, 32 * 14 * 14)  # -> [batch_size, 6272]
        x = self.fc(x)       # -> [batch_size, 10]
        return x