# Julia Image Classifier

```bash
image_classifier_julia/
├── src/
│   ├── data_loader.jl      # 数据加载和预处理
│   ├── layers.jl           # 神经网络层实现
│   ├── activation.jl       # 激活函数
│   ├── loss.jl            # 损失函数
│   ├── optimizer.jl       # 优化器
│   ├── network.jl         # 网络组装和前向传播
│   └── train.jl           # 训练循环
├── main.jl                # 主程序入口
└── utils.jl              # 工具函数
```

## 1. 数据加载与预处理模块

> dataLoader.jl

获取原始图像数据并转化为神经网络可处理的格式。

- 图像读取：从文件中加载图像文件
- 格式转换：将图像转换为数值矩阵
- 尺寸标准化：将所有图像调整为相同尺寸

> preprocessing.jl

对数据进行预处理。

- 数据增强：生成训练变体，增加数据多样性
- 批处理：将数据分成小批次，支持批量训练
- 标签编码：将类别标签转为数据标签

## 2
