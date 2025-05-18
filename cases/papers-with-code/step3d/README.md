# STEP3D

STEP文件特征识别和3D可视化工具。

## 功能特点

- STEP文件解析和几何提取
- 自动特征识别（孔、槽、凸台）
- 3D可视化与交互式显示
- 特征参数提取和分析

## 安装

### 使用Conda（推荐）

1. 创建并激活conda环境：
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate papers
```

2. 安装项目包：
```bash
# 在开发模式下安装
pip install -e .
```

### 使用UV（替代方案）

1. 首先安装基础依赖：
```bash
# 使用UV安装基础依赖
uv pip install -e .
```

2. 安装pythonocc-core：
```bash
# 使用requirements.txt安装pythonocc-core
uv pip install -r requirements.txt
```

## 使用方法

```python
from step3d.core import StepReader
from step3d.features import FeatureRecognizer
from step3d.visualization import Visualizer

# 读取STEP文件
reader = StepReader()
shape = reader.read_file("path/to/your/file.step")

# 识别特征
recognizer = FeatureRecognizer()
features = recognizer.recognize(shape)

# 可视化结果
visualizer = Visualizer()
visualizer.display(shape, features)
```

## 开发

1. 克隆仓库
2. 安装开发依赖：
   ```bash
   # 如果使用conda
   conda env create -f environment.yml
   conda activate step3d
   pip install -e .
   
   # 如果使用UV
   uv pip install -e ".[dev]"
   ```
3. 运行测试：
   ```bash
   pytest
   ```

## 许可证

MIT License 