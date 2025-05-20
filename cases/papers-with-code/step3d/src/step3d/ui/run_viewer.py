#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行STEP文件查看器
"""

import sys

try:
    # 初始化OCC显示后端
    from OCC.Display.backend import load_backend

    # 加载Qt后端
    load_backend("pyqt5")

    from PyQt5.QtWidgets import QApplication
    from step3d.ui.step_viewer import StepViewerWindow
except ImportError as e:
    print(f"\n错误: 无法加载必要的库: {e}")
    print("请确保已安装以下库:")
    print("  - PyQt5")
    print("  - pythonocc-core")
    print("  - OCC-Core")
    print("\n可以使用以下命令安装:")
    print("  pip install PyQt5 pythonocc-core")
    print("  或")
    print("  conda install -c conda-forge pythonocc-core pyqt")
    sys.exit(1)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = StepViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
