"""
STEP文件特征识别GUI界面
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QLabel
)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

from ..core import StepReader
from ..features import FeatureRecognizer
from .vtk_utils import create_vtk_actor_from_shape

class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP特征识别器")
        self.setMinimumSize(1200, 800)
        
        # 创建主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 文件选择按钮
        self.file_button = QPushButton("选择STEP文件")
        self.file_button.clicked.connect(self.open_file)
        left_layout.addWidget(self.file_button)
        
        # 特征树
        self.feature_tree = QTreeWidget()
        self.feature_tree.setHeaderLabel("特征列表")
        left_layout.addWidget(self.feature_tree)
        
        # 添加左侧面板到主布局
        layout.addWidget(left_panel, stretch=1)
        
        # 右侧3D视图
        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget, stretch=2)
        
        # 初始化VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # 设置背景色
        self.renderer.SetBackground(0.2, 0.3, 0.4)
        
        # 初始化工具类
        self.step_reader = StepReader()
        self.feature_recognizer = FeatureRecognizer()
        
        # 当前加载的形状
        self.current_shape = None
        
    def open_file(self):
        """打开STEP文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择STEP文件",
            "",
            "STEP Files (*.stp *.step);;All Files (*)"
        )
        
        if file_name:
            self.load_step_file(file_name)
    
    def load_step_file(self, file_path):
        """加载STEP文件并处理"""
        try:
            # 读取STEP文件
            self.current_shape = self.step_reader.read_file(file_path)
            
            # 识别特征
            features = self.feature_recognizer.recognize(self.current_shape)
            
            # 更新特征树
            self.update_feature_tree(features)
            
            # 更新3D视图
            self.update_3d_view()
            
        except Exception as e:
            print(f"错误：{str(e)}")
    
    def update_feature_tree(self, features):
        """更新特征树显示"""
        self.feature_tree.clear()
        
        # 添加孔特征
        holes_item = QTreeWidgetItem(["孔特征"])
        self.feature_tree.addTopLevelItem(holes_item)
        for hole in features['holes']:
            hole_info = f"半径: {hole['radius']:.2f}"
            item = QTreeWidgetItem([hole_info])
            holes_item.addChild(item)
        
        # 添加槽特征
        slots_item = QTreeWidgetItem(["槽特征"])
        self.feature_tree.addTopLevelItem(slots_item)
        for slot in features['slots']:
            slot_info = f"槽 {slot}"  # TODO: 添加更多槽信息
            item = QTreeWidgetItem([slot_info])
            slots_item.addChild(item)
        
        # 添加凸台特征
        bosses_item = QTreeWidgetItem(["凸台特征"])
        self.feature_tree.addTopLevelItem(bosses_item)
        for boss in features['bosses']:
            boss_info = f"凸台 {boss}"  # TODO: 添加更多凸台信息
            item = QTreeWidgetItem([boss_info])
            bosses_item.addChild(item)
        
        # 展开所有项
        self.feature_tree.expandAll()
    
    def update_3d_view(self):
        """更新3D视图显示"""
        if self.current_shape:
            # 清除现有的actors
            self.renderer.RemoveAllViewProps()
            
            # 创建新的actor
            actor = create_vtk_actor_from_shape(self.current_shape)
            self.renderer.AddActor(actor)
            
            # 重置相机
            self.renderer.ResetCamera()
            
            # 刷新显示
            self.vtk_widget.GetRenderWindow().Render()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.vtk_widget.Finalize()
        super().closeEvent(event) 