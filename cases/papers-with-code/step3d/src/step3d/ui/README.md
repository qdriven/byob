# STEP3D UI 模块

这个模块提供了一个图形用户界面，用于加载、查看和分析STEP文件。

## 功能

- 加载STEP文件
- 显示3D模型
- 识别和显示特征（孔、槽、凸台）
- 高亮显示选中的特征

## 运行UI

有几种方式可以运行UI：

### 方法1：直接运行Python模块

```bash
python -m step3d.ui.run_viewer
```

### 方法2：如果已安装为可执行程序

```bash
step-viewer
```

## 使用方法

1. 启动UI后，点击"文件" > "打开STEP文件"，选择一个STEP文件
2. 左侧面板将显示识别出的特征列表，包括孔、槽和凸台
3. 右侧面板将显示3D模型
4. 点击左侧的特征项可以在3D视图中高亮显示该特征
5. 使用菜单栏中的"视图"选项可以切换显示模式（线框/实体）或重置视图

## 依赖项

- PyQt5
- pythonocc-core (OCC-Core)

## 故障排除

### 检查环境

您可以运行环境检查脚本来确保所有必要的依赖项都已正确安装：

```bash
python -m step3d.ui.check_environment
```

### 安装依赖项

如果遇到导入错误，请确保已安装所有必要的依赖项：

```bash
# 使用pip安装
pip install PyQt5 pythonocc-core

# 或使用conda安装
conda install -c conda-forge pythonocc-core pyqt
```

### 常见错误

1. **后端字符串错误**

   如果遇到类似下面的错误：

   ```
   incompatible backend_str specified: qt-pyqt5
   backend is one of : ('pyqt5', 'pyqt6', 'pyside2', 'pyside6', 'wx', 'tk')
   ```

   请确保使用正确的后端字符串，应该是 `pyqt5` 而不是 `qt-pyqt5`。

2. **找不到模块**

   如果遇到类似下面的错误：

   ```
   ModuleNotFoundError: No module named 'OCC'
   ```

   请确保已正确安装 pythonocc-core 库。
