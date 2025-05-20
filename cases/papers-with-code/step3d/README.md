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

1. 使用 uv project 创建环境：
```bash
# 创建环境
uv venv

# 激活环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
# 安装基本依赖
uv pip install -r requirements.txt

# 如果是开发环境，安装开发依赖
uv pip install -r requirements-dev.txt
```

3. 安装项目（开发模式）：
```bash
uv pip install -e .
```

## 使用方法

### 图形界面

安装后，可以直接运行图形界面查看器：

```bash
# 检查环境是否正确设置
python -m step3d.ui.check_environment

# 运行图形界面
python -m step3d.ui.run_viewer
```

图形界面提供以下功能：

- 加载STEP文件
- 在左侧面板显示检测到的特征列表（孔、槽、凸台）
- 在右侧面板显示3D模型
- 点击左侧的特征项可以在3D视图中高亮显示该特征

### 编程接口

```python
from step3d.core import StepReader
from step3d.features import FeatureRecognizer

# 读取STEP文件
reader = StepReader()
shape = reader.read_file("path/to/your/file.step")

# 识别特征
recognizer = FeatureRecognizer()
features = recognizer.recognize(shape)

# 使用UI显示结果
from step3d.ui import StepViewerWindow
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
viewer = StepViewerWindow()
viewer.current_shape = shape
viewer.features = features
viewer.update_feature_tree()
viewer.display_shape(shape)
viewer.show()
sys.exit(app.exec_())
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
   uv venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate  # Windows
   uv pip install -r requirements-dev.txt
   uv pip install -e .
   ```
3. 运行测试：
   ```bash
   python -m pytest
   ```

4. 运行代码格式化：
   ```bash
   # 使用black格式化代码
   black src tests

   # 使用isort排序导入
   isort src tests
   ```

5. 运行类型检查：
   ```bash
   mypy src
   ```

## 许可证

MIT License