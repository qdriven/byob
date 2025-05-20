#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP文件查看器 - 用于加载STEP文件并显示检测到的特征
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QAction, QFileDialog, QTreeWidget, QTreeWidgetItem, QLabel,
        QSplitter, QStatusBar, QMessageBox
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon

    from OCC.Core.TopoDS import TopoDS_Shape

    # 初始化OCC显示后端
    from OCC.Display import OCCViewer
    from OCC.Display.backend import load_backend, get_qt_modules

    # 加载Qt后端
    load_backend("pyqt5")

    # 现在可以安全地导入qtViewer3d
    from OCC.Display.qtDisplay import qtViewer3d
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCone
    )
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

    from step3d.core import StepReader
    from step3d.features import FeatureRecognizer
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


class StepViewerWindow(QMainWindow):
    """STEP文件查看器主窗口"""

    def __init__(self):
        super().__init__()

        self.step_reader = StepReader()
        self.feature_recognizer = FeatureRecognizer()
        self.current_shape = None
        self.features = None

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle('STEP文件特征查看器')
        self.resize(1200, 800)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)

        # 创建左侧特征面板
        self.feature_tree = QTreeWidget()
        self.feature_tree.setHeaderLabels(['特征', '值'])
        self.feature_tree.setMinimumWidth(300)
        self.feature_tree.itemClicked.connect(self.on_feature_clicked)

        # 创建右侧3D视图
        self.display_widget = QWidget()
        self.display_layout = QVBoxLayout()
        self.display_widget.setLayout(self.display_layout)

        # 创建OCC 3D查看器
        self.canvas = qtViewer3d(self.display_widget)
        self.display_layout.addWidget(self.canvas)

        # 将部件添加到分割器
        splitter.addWidget(self.feature_tree)
        splitter.addWidget(self.display_widget)
        splitter.setSizes([300, 900])  # 设置初始分割比例

        # 创建主布局并添加分割器
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('就绪')

        # 初始化3D显示
        self.canvas.InitDriver()
        self.display = self.canvas._display

        # 设置显示模式
        self.display.SetSelectionModeVertex()  # 允许选择顶点

        # 显示坐标轴
        # 根据不同版本的PythonOCC，方法名可能不同
        try:
            # 新版本的API
            self.display.display_triedron()
        except AttributeError:
            try:
                # 旧版本的API
                self.display.DisplayTriedron()
            except AttributeError:
                print("\n警告: 无法显示坐标轴，跳过此步骤")

        # 设置背景颜色
        try:
            # 新版本的API
            self.display.set_bg_gradient_color([240, 240, 240], [255, 255, 255])
        except AttributeError:
            try:
                # 旧版本的API
                self.display.SetBackgroundColor([240/255, 240/255, 240/255])
            except AttributeError:
                print("\n警告: 无法设置背景颜色，跳过此步骤")

        # 初始化视图
        try:
            # 新版本的API
            self.display.View_Iso()
            self.display.FitAll()
        except AttributeError:
            try:
                # 旧版本的API
                self.display.set_view('iso')
                self.display.fit_all()
            except AttributeError:
                print("\n警告: 无法初始化视图，跳过此步骤")

    def create_menu_bar(self):
        """创建菜单栏"""
        # 创建菜单栏
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 打开文件动作
        open_action = QAction('打开STEP文件', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_step_file)
        file_menu.addAction(open_action)

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu('视图')

        # 重置视图动作
        reset_view_action = QAction('重置视图', self)
        reset_view_action.setShortcut('Ctrl+R')
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

        # 显示线框动作
        wireframe_action = QAction('线框模式', self)
        wireframe_action.setShortcut('Ctrl+W')
        wireframe_action.triggered.connect(self.set_wireframe_mode)
        view_menu.addAction(wireframe_action)

        # 显示实体动作
        shaded_action = QAction('实体模式', self)
        shaded_action.setShortcut('Ctrl+S')
        shaded_action.triggered.connect(self.set_shaded_mode)
        view_menu.addAction(shaded_action)

    def open_step_file(self):
        """打开STEP文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开STEP文件", "",
            "STEP Files (*.step *.stp);;All Files (*)",
            options=options
        )

        if file_path:
            self.statusBar.showMessage(f'正在加载文件: {file_path}')
            try:
                # 加载STEP文件
                shape = self.step_reader.read_file(file_path)
                self.current_shape = shape

                # 识别特征
                self.features = self.feature_recognizer.recognize(shape)

                # 更新UI
                self.update_feature_tree()
                self.display_shape(shape)

                # 显示文件名
                file_name = os.path.basename(file_path)
                self.statusBar.showMessage(f'已加载文件: {file_name}')

            except Exception as e:
                self.statusBar.showMessage(f'加载文件失败: {str(e)}')
                QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")

    def update_feature_tree(self):
        """更新特征树"""
        self.feature_tree.clear()

        if not self.features:
            return

        # 创建根节点
        root = QTreeWidgetItem(self.feature_tree, ["STEP文件特征"])
        root.setExpanded(True)

        # 添加孔特征
        if 'holes' in self.features and self.features['holes']:
            holes_item = QTreeWidgetItem(root, [f"孔 ({len(self.features['holes'])})"])
            holes_item.setExpanded(True)

            for i, hole in enumerate(self.features['holes']):
                hole_item = QTreeWidgetItem(holes_item, [f"孔 #{i+1}", ""])
                hole_item.setData(0, Qt.UserRole, ('hole', i))

                # 添加孔的属性
                self.add_feature_properties(hole_item, hole)

        # 添加槽特征
        if 'slots' in self.features and self.features['slots']:
            slots_item = QTreeWidgetItem(root, [f"槽 ({len(self.features['slots'])})"])
            slots_item.setExpanded(True)

            for i, slot in enumerate(self.features['slots']):
                slot_item = QTreeWidgetItem(slots_item, [f"槽 #{i+1}", ""])
                slot_item.setData(0, Qt.UserRole, ('slot', i))

                # 添加槽的属性
                self.add_feature_properties(slot_item, slot)

        # 添加凸台特征
        if 'bosses' in self.features and self.features['bosses']:
            bosses_item = QTreeWidgetItem(root, [f"凸台 ({len(self.features['bosses'])})"])
            bosses_item.setExpanded(True)

            for i, boss in enumerate(self.features['bosses']):
                boss_item = QTreeWidgetItem(bosses_item, [f"凸台 #{i+1}", ""])
                boss_item.setData(0, Qt.UserRole, ('boss', i))

                # 添加凸台的属性
                self.add_feature_properties(boss_item, boss)

    def add_feature_properties(self, parent_item, feature):
        """添加特征属性到树节点"""
        for key, value in feature.items():
            if key == 'location':
                # 位置信息特殊处理
                loc_item = QTreeWidgetItem(parent_item, ["位置", ""])
                x, y, z = feature['location']
                QTreeWidgetItem(loc_item, ["X", f"{x:.2f}"])
                QTreeWidgetItem(loc_item, ["Y", f"{y:.2f}"])
                QTreeWidgetItem(loc_item, ["Z", f"{z:.2f}"])
            elif key == 'axis' and isinstance(value, tuple):
                # 轴向信息特殊处理
                axis_item = QTreeWidgetItem(parent_item, ["轴向", ""])
                dx, dy, dz = feature['axis']
                QTreeWidgetItem(axis_item, ["X", f"{dx:.2f}"])
                QTreeWidgetItem(axis_item, ["Y", f"{dy:.2f}"])
                QTreeWidgetItem(axis_item, ["Z", f"{dz:.2f}"])
            elif key == 'direction' and isinstance(value, tuple):
                # 方向信息特殊处理
                dir_item = QTreeWidgetItem(parent_item, ["方向", ""])
                dx, dy, dz = feature['direction']
                QTreeWidgetItem(dir_item, ["X", f"{dx:.2f}"])
                QTreeWidgetItem(dir_item, ["Y", f"{dy:.2f}"])
                QTreeWidgetItem(dir_item, ["Z", f"{dz:.2f}"])
            else:
                # 其他属性
                if isinstance(value, (int, float)):
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    QTreeWidgetItem(parent_item, [self.get_property_name(key), value_str])
                elif isinstance(value, str):
                    QTreeWidgetItem(parent_item, [self.get_property_name(key), value])

    def get_property_name(self, key):
        """获取属性的中文名称"""
        property_names = {
            'type': '类型',
            'radius': '半径',
            'diameter': '直径',
            'depth': '深度',
            'width': '宽度',
            'length': '长度',
            'height': '高度',
            'angle': '角度',
            'top_radius': '顶部半径',
            'bottom_radius': '底部半径',
            'top_diameter': '顶部直径',
            'bottom_diameter': '底部直径'
        }
        return property_names.get(key, key)

    def display_shape(self, shape):
        """显示形状"""
        try:
            # 新版本的API
            self.display.EraseAll()

            # 显示原始形状
            self.display.DisplayShape(shape, update=True)
        except AttributeError:
            try:
                # 旧版本的API
                self.display.erase_all()

                # 显示原始形状
                self.display.display_shape(shape, update=True)
            except AttributeError:
                print("\n警告: 无法显示形状，跳过此步骤")

        # 重置视图
        self.reset_view()

    def display_reconstructed_model(self):
        """显示重建的模型"""
        if not self.features:
            return

        self.display.EraseAll()

        # 创建基础形状（简单的盒子）
        base_shape = self.create_base_shape()

        # 添加孔（通过从基础形状中减去圆柱体）
        if 'holes' in self.features:
            for hole in self.features['holes']:
                base_shape = self.add_hole_to_shape(base_shape, hole)

        # 添加槽（通过从基础形状中减去长方体）
        if 'slots' in self.features:
            for slot in self.features['slots']:
                base_shape = self.add_slot_to_shape(base_shape, slot)

        # 添加凸台（通过向基础形状添加圆柱体）
        if 'bosses' in self.features:
            for boss in self.features['bosses']:
                base_shape = self.add_boss_to_shape(base_shape, boss)

        # 显示重建的形状
        self.display.DisplayShape(base_shape, update=True)

        # 重置视图
        self.reset_view()

    def create_base_shape(self):
        """创建基础形状"""
        # 创建一个简单的盒子作为基础形状
        # 尺寸可以根据特征的位置和大小来确定
        return BRepPrimAPI_MakeBox(100, 100, 20).Shape()

    def add_hole_to_shape(self, base_shape, hole):
        """向形状添加孔"""
        # 获取孔的参数
        x, y, z = hole['location']
        radius = hole.get('radius', 5.0)
        depth = hole.get('depth', 20.0)

        # 创建圆柱体表示孔
        if 'axis' in hole:
            dx, dy, dz = hole['axis']
            axis = gp_Ax2(gp_Pnt(x, y, z), gp_Dir(dx, dy, dz))
        else:
            axis = gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1))

        cylinder = BRepPrimAPI_MakeCylinder(axis, radius, depth).Shape()

        # 从基础形状中减去圆柱体
        return BRepAlgoAPI_Cut(base_shape, cylinder).Shape()

    def add_slot_to_shape(self, base_shape, slot):
        """向形状添加槽"""
        # 获取槽的参数
        x, y, z = slot['location']
        width = slot.get('width', 10.0)
        length = slot.get('length', 20.0)
        depth = slot.get('depth', 10.0)

        # 创建长方体表示槽
        box = BRepPrimAPI_MakeBox(gp_Pnt(x - length/2, y - width/2, z), length, width, depth).Shape()

        # 从基础形状中减去长方体
        return BRepAlgoAPI_Cut(base_shape, box).Shape()

    def add_boss_to_shape(self, base_shape, boss):
        """向形状添加凸台"""
        # 获取凸台的参数
        x, y, z = boss['location']
        height = boss.get('height', 10.0)

        if 'diameter' in boss:
            # 圆柱形凸台
            diameter = boss['diameter']
            radius = diameter / 2.0
            cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)), radius, height).Shape()
            return BRepAlgoAPI_Fuse(base_shape, cylinder).Shape()
        elif 'width' in boss and 'length' in boss:
            # 矩形凸台
            width = boss['width']
            length = boss['length']
            box = BRepPrimAPI_MakeBox(gp_Pnt(x - length/2, y - width/2, z), length, width, height).Shape()
            return BRepAlgoAPI_Fuse(base_shape, box).Shape()

        return base_shape

    def on_feature_clicked(self, item, column):
        """特征项被点击时的处理"""
        # 获取特征数据
        feature_data = item.data(0, Qt.UserRole)
        if not feature_data:
            return

        feature_type, index = feature_data

        # 高亮显示选中的特征
        self.highlight_feature(feature_type, index)

    def highlight_feature(self, feature_type, index):
        """高亮显示特征"""
        if not self.features:
            return

        # 重新显示原始形状
        try:
            # 新版本的API
            self.display.EraseAll()
            self.display.DisplayShape(self.current_shape, update=False)
        except AttributeError:
            try:
                # 旧版本的API
                self.display.erase_all()
                self.display.display_shape(self.current_shape, update=False)
            except AttributeError:
                print("\n警告: 无法高亮显示特征，跳过此步骤")
                return

        # 高亮显示选中的特征
        if feature_type == 'hole' and 'holes' in self.features and index < len(self.features['holes']):
            hole = self.features['holes'][index]
            self.highlight_hole(hole)
        elif feature_type == 'slot' and 'slots' in self.features and index < len(self.features['slots']):
            slot = self.features['slots'][index]
            self.highlight_slot(slot)
        elif feature_type == 'boss' and 'bosses' in self.features and index < len(self.features['bosses']):
            boss = self.features['bosses'][index]
            self.highlight_boss(boss)

        self.display.Repaint()

    def highlight_hole(self, hole):
        """高亮显示孔"""
        # 获取孔的参数
        x, y, z = hole['location']
        radius = hole.get('radius', 5.0)

        # 创建一个点表示孔的位置
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()
        try:
            # 新版本的API
            self.display.DisplayShape(vertex, update=False, color=Quantity_Color(1, 0, 0, Quantity_TOC_RGB))
        except AttributeError:
            try:
                # 旧版本的API
                self.display.display_shape(vertex, update=False, color=(1, 0, 0))
            except AttributeError:
                print("\n警告: 无法高亮显示孔，跳过此步骤")

        # 如果有轴向，显示轴向
        if 'axis' in hole:
            dx, dy, dz = hole['axis']
            length = 20.0  # 轴向线的长度
            self.display_arrow(x, y, z, dx, dy, dz, length)

    def highlight_slot(self, slot):
        """高亮显示槽"""
        # 获取槽的参数
        x, y, z = slot['location']

        # 创建一个点表示槽的位置
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()
        try:
            # 新版本的API
            self.display.DisplayShape(vertex, update=False, color=Quantity_Color(0, 1, 0, Quantity_TOC_RGB))
        except AttributeError:
            try:
                # 旧版本的API
                self.display.display_shape(vertex, update=False, color=(0, 1, 0))
            except AttributeError:
                print("\n警告: 无法高亮显示槽，跳过此步骤")

        # 如果有方向，显示方向
        if 'direction' in slot:
            dx, dy, dz = slot['direction']
            length = 20.0  # 方向线的长度
            self.display_arrow(x, y, z, dx, dy, dz, length)

    def highlight_boss(self, boss):
        """高亮显示凸台"""
        # 获取凸台的参数
        x, y, z = boss['location']

        # 创建一个点表示凸台的位置
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()
        try:
            # 新版本的API
            self.display.DisplayShape(vertex, update=False, color=Quantity_Color(0, 0, 1, Quantity_TOC_RGB))
        except AttributeError:
            try:
                # 旧版本的API
                self.display.display_shape(vertex, update=False, color=(0, 0, 1))
            except AttributeError:
                print("\n警告: 无法高亮显示凸台，跳过此步骤")

    def display_arrow(self, x, y, z, dx, dy, dz, length):
        """显示箭头（方向线）"""
        # 创建起点和终点
        start_point = gp_Pnt(x, y, z)
        end_point = gp_Pnt(x + dx * length, y + dy * length, z + dz * length)

        # 显示线段
        try:
            # 新版本的API
            self.display.DisplayVector(gp_Vec(dx, dy, dz), start_point, update=False)
        except AttributeError:
            try:
                # 旧版本的API
                self.display.display_vector(gp_Vec(dx, dy, dz), start_point, update=False)
            except AttributeError:
                try:
                    # 如果上述方法都不可用，尝试使用线段
                    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                    edge = BRepBuilderAPI_MakeEdge(start_point, end_point).Edge()
                    self.display.DisplayShape(edge, update=False, color=Quantity_Color(1, 0, 0, Quantity_TOC_RGB))
                except Exception as e:
                    print(f"\n警告: 无法显示箭头: {e}")

    def reset_view(self):
        """重置视图"""
        try:
            # 新版本的API
            self.display.View_Iso()
            self.display.FitAll()
        except AttributeError:
            try:
                # 旧版本的API
                self.display.set_view('iso')
                self.display.fit_all()
            except AttributeError:
                print("\n警告: 无法重置视图，跳过此步骤")

    def set_wireframe_mode(self):
        """设置线框模式"""
        try:
            # 新版本的API
            self.display.SetModeWireFrame()
        except AttributeError:
            try:
                # 旧版本的API
                self.display.set_display_mode(0)  # 0 = wireframe
            except AttributeError:
                print("\n警告: 无法设置线框模式，跳过此步骤")

    def set_shaded_mode(self):
        """设置实体模式"""
        try:
            # 新版本的API
            self.display.SetModeShaded()
        except AttributeError:
            try:
                # 旧版本的API
                self.display.set_display_mode(1)  # 1 = shaded
            except AttributeError:
                print("\n警告: 无法设置实体模式，跳过此步骤")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = StepViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
