#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查环境是否正确设置
"""

import sys
import importlib.util


def check_module(module_name):
    """检查模块是否已安装"""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} 未安装")
        return False
    else:
        print(f"✅ {module_name} 已安装")
        return True


def check_environment():
    """检查环境是否正确设置"""
    print("检查环境...")

    # 检查必要的模块
    modules = [
        "PyQt5",
        "OCC.Core",
        "OCC.Display",
        "numpy",
    ]

    all_installed = True
    for module in modules:
        if not check_module(module):
            all_installed = False

    # 检查PyQt5后端
    if all_installed:
        try:
            from OCC.Display.backend import load_backend
            load_backend("pyqt5")
            print("✅ PyQt5后端加载成功")

            # 检查PythonOCC版本
            import OCC
            print(f"✅ PythonOCC版本: {OCC.__version__ if hasattr(OCC, '__version__') else '\u672a知'}")

            # 检查API兼容性
            from OCC.Display.qtDisplay import qtViewer3d
            viewer = qtViewer3d(None)
            display = viewer._display

            # 检查方法名称
            api_methods = {
                '显示坐标轴': hasattr(display, 'display_triedron') or hasattr(display, 'DisplayTriedron'),
                '设置背景颜色': hasattr(display, 'set_bg_gradient_color') or hasattr(display, 'SetBackgroundColor'),
                '设置视图': hasattr(display, 'set_view') or hasattr(display, 'View_Iso'),
                '显示形状': hasattr(display, 'display_shape') or hasattr(display, 'DisplayShape'),
            }

            for method_name, available in api_methods.items():
                if available:
                    print(f"✅ API方法 '{method_name}' 可用")
                else:
                    print(f"❌ API方法 '{method_name}' 不可用")
                    all_installed = False
        except Exception as e:
            print(f"❌ PyQt5后端加载失败: {e}")
            all_installed = False

    # 检查OCC.Display.qtDisplay
    if all_installed:
        try:
            from OCC.Display.qtDisplay import qtViewer3d
            print("✅ OCC.Display.qtDisplay加载成功")
        except Exception as e:
            print(f"❌ OCC.Display.qtDisplay加载失败: {e}")
            all_installed = False

    # 总结
    if all_installed:
        print("\n✅ 环境检查通过，所有必要的模块都已安装")
        print("  可以运行UI: python -m step3d.ui.run_viewer")
    else:
        print("\n❌ 环境检查失败，请安装缺少的模块")
        print("  可以使用以下命令安装依赖:")
        print("  pip install PyQt5 pythonocc-core numpy")
        print("  或")
        print("  conda install -c conda-forge pythonocc-core pyqt numpy")


if __name__ == "__main__":
    check_environment()
