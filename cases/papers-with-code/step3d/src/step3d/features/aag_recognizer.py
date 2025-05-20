#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于属性邻接图(AAG)的特征识别器
"""

import math
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
import networkx as nx

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
    GeomAbs_Line, GeomAbs_Circle
)
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepTools import breptools

# 面类型枚举
SURFACE_TYPES = {
    GeomAbs_Plane: "PLANE",
    GeomAbs_Cylinder: "CYLINDER",
    GeomAbs_Cone: "CONE",
    GeomAbs_Sphere: "SPHERE",
    GeomAbs_Torus: "TORUS",
    GeomAbs_BezierSurface: "BEZIER",
    GeomAbs_BSplineSurface: "BSPLINE",
    GeomAbs_SurfaceOfRevolution: "REVOLUTION",
    GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
    GeomAbs_OffsetSurface: "OFFSET"
}

# 边类型枚举
CURVE_TYPES = {
    GeomAbs_Line: "LINE",
    GeomAbs_Circle: "CIRCLE"
}

# 凹凸性枚举
CONCAVITY_TYPES = {
    "CONCAVE": -1,  # 凹
    "CONVEX": 1,    # 凸
    "SMOOTH": 0     # 平滑
}


class AAGNode:
    """AAG图中的节点，表示一个面"""
    
    def __init__(self, face_id: int, face: TopoDS_Face):
        self.id = face_id
        self.face = face
        self.surface_type = None
        self.normal = None
        self.center = None
        self.area = None
        self.neighbors = set()  # 相邻面的ID集合
        self.concavity = {}     # 与相邻面的凹凸性关系，{neighbor_id: concavity_type}
        
        self._analyze_face()
    
    def _analyze_face(self):
        """分析面的属性"""
        # 获取面的几何类型
        surface = BRepAdaptor_Surface(self.face)
        self.surface_type = surface.GetType()
        
        # 计算面的法向量和中心点
        props = GProp_GProps()
        brepgprop.SurfaceProperties(self.face, props)
        self.area = props.Mass()
        self.center = props.CentreOfMass()
        
        # 根据面的类型获取法向量
        if self.surface_type == GeomAbs_Plane:
            self.normal = surface.Plane().Axis().Direction()
        elif self.surface_type == GeomAbs_Cylinder:
            # 对于圆柱面，法向量是从轴向到表面点的径向向量
            cylinder = surface.Cylinder()
            axis = cylinder.Axis().Location()
            # 计算从轴到中心点的向量
            self.normal = gp_Dir(
                self.center.X() - axis.X(),
                self.center.Y() - axis.Y(),
                self.center.Z() - axis.Z()
            )
    
    def add_neighbor(self, neighbor_id: int, concavity: int):
        """添加相邻面"""
        self.neighbors.add(neighbor_id)
        self.concavity[neighbor_id] = concavity
    
    def is_neighbor(self, face_id: int) -> bool:
        """检查是否是相邻面"""
        return face_id in self.neighbors
    
    def get_concavity(self, neighbor_id: int) -> int:
        """获取与相邻面的凹凸性关系"""
        return self.concavity.get(neighbor_id, 0)
    
    def get_surface_type_name(self) -> str:
        """获取面类型的名称"""
        return SURFACE_TYPES.get(self.surface_type, "UNKNOWN")
    
    def __str__(self) -> str:
        return f"Face {self.id} ({self.get_surface_type_name()})"


class AAGEdge:
    """AAG图中的边，表示两个面之间的关系"""
    
    def __init__(self, face1_id: int, face2_id: int, edge: TopoDS_Shape, concavity: int):
        self.face1_id = face1_id
        self.face2_id = face2_id
        self.edge = edge
        self.concavity = concavity
        self.edge_type = None
        self.length = None
        
        self._analyze_edge()
    
    def _analyze_edge(self):
        """分析边的属性"""
        # 获取边的几何类型
        curve_adaptor = BRepAdaptor_Curve(topods.Edge(self.edge))
        self.edge_type = curve_adaptor.GetType()
        
        # 计算边的长度
        props = GProp_GProps()
        brepgprop.LinearProperties(self.edge, props)
        self.length = props.Mass()
    
    def get_edge_type_name(self) -> str:
        """获取边类型的名称"""
        return CURVE_TYPES.get(self.edge_type, "UNKNOWN")
    
    def __str__(self) -> str:
        concavity_str = "CONCAVE" if self.concavity == -1 else "CONVEX" if self.concavity == 1 else "SMOOTH"
        return f"Edge {self.face1_id}-{self.face2_id} ({self.get_edge_type_name()}, {concavity_str})"


class AAGGraph:
    """属性邻接图(AAG)"""
    
    def __init__(self, shape: TopoDS_Shape):
        self.shape = shape
        self.nodes = {}  # {face_id: AAGNode}
        self.edges = []  # [AAGEdge]
        self.graph = nx.Graph()  # NetworkX图用于算法分析
        
        self._build_graph()
    
    def _build_graph(self):
        """构建AAG图"""
        # 1. 收集所有面
        face_map = TopTools_IndexedMapOfShape()
        explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        face_id = 1
        
        while explorer.More():
            face = topods.Face(explorer.Current())
            self.nodes[face_id] = AAGNode(face_id, face)
            self.graph.add_node(face_id, face=face)
            face_id += 1
            explorer.Next()
        
        # 2. 分析面之间的关系
        for i in range(1, face_id):
            for j in range(i+1, face_id):
                if self._are_faces_adjacent(self.nodes[i].face, self.nodes[j].face):
                    # 计算凹凸性
                    concavity = self._calculate_concavity(self.nodes[i], self.nodes[j])
                    
                    # 添加邻接关系
                    self.nodes[i].add_neighbor(j, concavity)
                    self.nodes[j].add_neighbor(i, concavity)
                    
                    # 添加边
                    edge = self._find_common_edge(self.nodes[i].face, self.nodes[j].face)
                    if edge:
                        aag_edge = AAGEdge(i, j, edge, concavity)
                        self.edges.append(aag_edge)
                    
                    # 更新NetworkX图
                    self.graph.add_edge(i, j, concavity=concavity)
    
    def _are_faces_adjacent(self, face1: TopoDS_Face, face2: TopoDS_Face) -> bool:
        """检查两个面是否相邻"""
        # 获取face1的所有边
        edges1 = set()
        explorer = TopExp_Explorer(face1, TopAbs_EDGE)
        while explorer.More():
            edges1.add(explorer.Current())
            explorer.Next()
        
        # 检查face2是否包含face1的任何边
        explorer = TopExp_Explorer(face2, TopAbs_EDGE)
        while explorer.More():
            if explorer.Current() in edges1:
                return True
            explorer.Next()
        
        return False
    
    def _find_common_edge(self, face1: TopoDS_Face, face2: TopoDS_Face) -> Optional[TopoDS_Shape]:
        """找到两个面之间的公共边"""
        # 获取face1的所有边
        edges1 = {}
        explorer = TopExp_Explorer(face1, TopAbs_EDGE)
        while explorer.More():
            edges1[explorer.Current()] = True
            explorer.Next()
        
        # 检查face2的边是否在face1中
        explorer = TopExp_Explorer(face2, TopAbs_EDGE)
        while explorer.More():
            if explorer.Current() in edges1:
                return explorer.Current()
            explorer.Next()
        
        return None
    
    def _calculate_concavity(self, node1: AAGNode, node2: AAGNode) -> int:
        """计算两个面之间的凹凸性关系"""
        # 如果任一面没有法向量，则返回平滑
        if not node1.normal or not node2.normal:
            return CONCAVITY_TYPES["SMOOTH"]
        
        # 计算从node1中心到node2中心的向量
        vec = gp_Vec(
            node2.center.X() - node1.center.X(),
            node2.center.Y() - node1.center.Y(),
            node2.center.Z() - node1.center.Z()
        )
        
        # 如果向量长度接近0，则返回平滑
        if vec.Magnitude() < 1e-6:
            return CONCAVITY_TYPES["SMOOTH"]
        
        # 计算node1法向量与向量的点积
        dot1 = node1.normal.X() * vec.X() + node1.normal.Y() * vec.Y() + node1.normal.Z() * vec.Z()
        
        # 计算node2法向量与向量的点积
        dot2 = node2.normal.X() * (-vec.X()) + node2.normal.Y() * (-vec.Y()) + node2.normal.Z() * (-vec.Z())
        
        # 如果两个点积都是正的，则是凹的
        if dot1 > 0 and dot2 > 0:
            return CONCAVITY_TYPES["CONCAVE"]
        # 如果两个点积都是负的，则是凸的
        elif dot1 < 0 and dot2 < 0:
            return CONCAVITY_TYPES["CONVEX"]
        # 否则是平滑的
        else:
            return CONCAVITY_TYPES["SMOOTH"]
    
    def find_subgraphs(self, condition) -> List[Set[int]]:
        """查找满足特定条件的子图"""
        # 创建一个新图，只包含满足条件的边
        filtered_graph = nx.Graph()
        
        for node_id in self.graph.nodes():
            filtered_graph.add_node(node_id)
        
        for u, v, data in self.graph.edges(data=True):
            if condition(self.nodes[u], self.nodes[v], data):
                filtered_graph.add_edge(u, v)
        
        # 查找连通分量
        return list(nx.connected_components(filtered_graph))
    
    def get_node(self, face_id: int) -> AAGNode:
        """获取节点"""
        return self.nodes.get(face_id)
    
    def get_nodes(self) -> Dict[int, AAGNode]:
        """获取所有节点"""
        return self.nodes
    
    def get_edges(self) -> List[AAGEdge]:
        """获取所有边"""
        return self.edges
    
    def get_face_ids(self) -> List[int]:
        """获取所有面ID"""
        return list(self.nodes.keys())
    
    def __str__(self) -> str:
        return f"AAG Graph with {len(self.nodes)} nodes and {len(self.edges)} edges"


class AAGFeatureRecognizer:
    """基于AAG的特征识别器"""
    
    def __init__(self):
        self.tolerance = 1e-6
    
    def recognize(self, shape: TopoDS_Shape) -> Dict[str, List[Dict[str, Any]]]:
        """识别形状中的特征"""
        # 构建AAG图
        aag = AAGGraph(shape)
        
        # 识别特征
        features = {
            'holes': self._recognize_holes(aag),
            'slots': self._recognize_slots(aag),
            'bosses': self._recognize_bosses(aag)
        }
        
        return features
    
    def _recognize_holes(self, aag: AAGGraph) -> List[Dict[str, Any]]:
        """识别孔特征"""
        holes = []
        
        # 查找满足孔特征条件的子图
        hole_subgraphs = aag.find_subgraphs(self._is_hole_condition)
        
        for subgraph in hole_subgraphs:
            # 分析子图，提取孔特征参数
            hole_params = self._extract_hole_parameters(aag, subgraph)
            if hole_params:
                holes.append(hole_params)
        
        return holes
    
    def _is_hole_condition(self, node1: AAGNode, node2: AAGNode, edge_data: Dict) -> bool:
        """检查两个面之间的关系是否满足孔特征条件"""
        # 孔特征条件：
        # 1. 至少一个面是圆柱面
        # 2. 关系是凹的
        is_cylinder1 = node1.surface_type == GeomAbs_Cylinder
        is_cylinder2 = node2.surface_type == GeomAbs_Cylinder
        is_concave = edge_data.get('concavity', 0) == CONCAVITY_TYPES["CONCAVE"]
        
        return (is_cylinder1 or is_cylinder2) and is_concave
    
    def _extract_hole_parameters(self, aag: AAGGraph, face_ids: Set[int]) -> Optional[Dict[str, Any]]:
        """从孔特征子图中提取参数"""
        # 查找圆柱面
        cylinder_faces = []
        plane_faces = []
        
        for face_id in face_ids:
            node = aag.get_node(face_id)
            if node.surface_type == GeomAbs_Cylinder:
                cylinder_faces.append(node)
            elif node.surface_type == GeomAbs_Plane:
                plane_faces.append(node)
        
        if not cylinder_faces:
            return None
        
        # 使用第一个圆柱面作为参考
        cylinder_node = cylinder_faces[0]
        cylinder_surface = BRepAdaptor_Surface(cylinder_node.face)
        cylinder = cylinder_surface.Cylinder()
        
        # 提取参数
        radius = cylinder.Radius()
        axis = cylinder.Axis()
        location = axis.Location()
        direction = axis.Direction()
        
        # 计算深度
        depth = 0
        if plane_faces:
            # 使用平面之间的距离作为深度
            if len(plane_faces) >= 2:
                plane1 = BRepAdaptor_Surface(plane_faces[0].face).Plane()
                plane2 = BRepAdaptor_Surface(plane_faces[1].face).Plane()
                depth = abs(plane1.Distance(plane2.Location()))
            else:
                # 使用圆柱高度作为深度
                props = GProp_GProps()
                brepgprop.LinearProperties(cylinder_node.face, props)
                depth = props.Mass() / (2 * math.pi * radius)
        
        # 确定孔的类型
        hole_type = "through" if len(plane_faces) >= 2 else "blind"
        
        return {
            'type': hole_type,
            'radius': radius,
            'location': (location.X(), location.Y(), location.Z()),
            'axis': (direction.X(), direction.Y(), direction.Z()),
            'depth': depth
        }
    
    def _recognize_slots(self, aag: AAGGraph) -> List[Dict[str, Any]]:
        """识别槽特征"""
        slots = []
        
        # 查找满足槽特征条件的子图
        slot_subgraphs = aag.find_subgraphs(self._is_slot_condition)
        
        for subgraph in slot_subgraphs:
            # 分析子图，提取槽特征参数
            slot_params = self._extract_slot_parameters(aag, subgraph)
            if slot_params:
                slots.append(slot_params)
        
        return slots
    
    def _is_slot_condition(self, node1: AAGNode, node2: AAGNode, edge_data: Dict) -> bool:
        """检查两个面之间的关系是否满足槽特征条件"""
        # 槽特征条件：
        # 1. 至少一个面是平面
        # 2. 关系是凹的
        is_plane1 = node1.surface_type == GeomAbs_Plane
        is_plane2 = node2.surface_type == GeomAbs_Plane
        is_concave = edge_data.get('concavity', 0) == CONCAVITY_TYPES["CONCAVE"]
        
        return (is_plane1 or is_plane2) and is_concave
    
    def _extract_slot_parameters(self, aag: AAGGraph, face_ids: Set[int]) -> Optional[Dict[str, Any]]:
        """从槽特征子图中提取参数"""
        # 查找平面
        plane_faces = []
        side_faces = []
        
        for face_id in face_ids:
            node = aag.get_node(face_id)
            if node.surface_type == GeomAbs_Plane:
                # 检查法向量方向，区分底面和侧面
                if node.normal and abs(node.normal.Z()) > 0.9:  # 近似垂直
                    plane_faces.append(node)
                else:
                    side_faces.append(node)
        
        if not plane_faces or len(side_faces) < 2:
            return None
        
        # 使用第一个平面作为参考
        bottom_node = plane_faces[0]
        
        # 计算槽的参数
        # 使用底面的边界框作为宽度和长度
        bounds = breptools.Bounds(bottom_node.face)
        width = bounds[1] - bounds[0]  # dx
        length = bounds[3] - bounds[2]  # dy
        
        # 计算深度
        depth = 0
        if len(plane_faces) >= 2:
            # 使用平面之间的距离作为深度
            plane1 = BRepAdaptor_Surface(plane_faces[0].face).Plane()
            plane2 = BRepAdaptor_Surface(plane_faces[1].face).Plane()
            depth = abs(plane1.Distance(plane2.Location()))
        else:
            # 使用侧面高度作为深度
            side_node = side_faces[0]
            props = GProp_GProps()
            brepgprop.LinearProperties(side_node.face, props)
            depth = props.Mass() / max(width, length)
        
        # 确定槽的类型
        slot_type = "rectangular"  # 默认为矩形槽
        
        # 确定方向
        direction = (1.0, 0.0, 0.0)  # 默认方向
        if bottom_node.normal:
            direction = (bottom_node.normal.X(), bottom_node.normal.Y(), bottom_node.normal.Z())
        
        return {
            'type': slot_type,
            'width': width,
            'length': length,
            'depth': depth,
            'location': (bottom_node.center.X(), bottom_node.center.Y(), bottom_node.center.Z()),
            'direction': direction
        }
    
    def _recognize_bosses(self, aag: AAGGraph) -> List[Dict[str, Any]]:
        """识别凸台特征"""
        bosses = []
        
        # 查找满足凸台特征条件的子图
        boss_subgraphs = aag.find_subgraphs(self._is_boss_condition)
        
        for subgraph in boss_subgraphs:
            # 分析子图，提取凸台特征参数
            boss_params = self._extract_boss_parameters(aag, subgraph)
            if boss_params:
                bosses.append(boss_params)
        
        return bosses
    
    def _is_boss_condition(self, node1: AAGNode, node2: AAGNode, edge_data: Dict) -> bool:
        """检查两个面之间的关系是否满足凸台特征条件"""
        # 凸台特征条件：
        # 1. 至少一个面是平面或圆柱面
        # 2. 关系是凸的
        is_plane1 = node1.surface_type == GeomAbs_Plane
        is_plane2 = node2.surface_type == GeomAbs_Plane
        is_cylinder1 = node1.surface_type == GeomAbs_Cylinder
        is_cylinder2 = node2.surface_type == GeomAbs_Cylinder
        is_convex = edge_data.get('concavity', 0) == CONCAVITY_TYPES["CONVEX"]
        
        return ((is_plane1 or is_plane2) or (is_cylinder1 or is_cylinder2)) and is_convex
    
    def _extract_boss_parameters(self, aag: AAGGraph, face_ids: Set[int]) -> Optional[Dict[str, Any]]:
        """从凸台特征子图中提取参数"""
        # 查找平面和圆柱面
        plane_faces = []
        cylinder_faces = []
        
        for face_id in face_ids:
            node = aag.get_node(face_id)
            if node.surface_type == GeomAbs_Plane:
                plane_faces.append(node)
            elif node.surface_type == GeomAbs_Cylinder:
                cylinder_faces.append(node)
        
        if not plane_faces:
            return None
        
        # 使用第一个平面作为参考
        top_node = plane_faces[0]
        
        # 确定凸台类型和参数
        if cylinder_faces:
            # 圆柱形凸台
            cylinder_node = cylinder_faces[0]
            cylinder_surface = BRepAdaptor_Surface(cylinder_node.face)
            cylinder = cylinder_surface.Cylinder()
            
            # 提取参数
            diameter = 2 * cylinder.Radius()
            axis = cylinder.Axis()
            location = axis.Location()
            
            # 计算高度
            height = 0
            if len(plane_faces) >= 2:
                # 使用平面之间的距离作为高度
                plane1 = BRepAdaptor_Surface(plane_faces[0].face).Plane()
                plane2 = BRepAdaptor_Surface(plane_faces[1].face).Plane()
                height = abs(plane1.Distance(plane2.Location()))
            else:
                # 使用圆柱高度作为高度
                props = GProp_GProps()
                brepgprop.LinearProperties(cylinder_node.face, props)
                height = props.Mass() / (2 * math.pi * cylinder.Radius())
            
            return {
                'type': 'cylindrical',
                'diameter': diameter,
                'height': height,
                'location': (location.X(), location.Y(), location.Z())
            }
        else:
            # 矩形凸台
            # 计算凸台的参数
            # 使用顶面的边界框作为宽度和长度
            bounds = breptools.Bounds(top_node.face)
            width = bounds[1] - bounds[0]  # dx
            length = bounds[3] - bounds[2]  # dy
            
            # 计算高度
            height = 0
            if len(plane_faces) >= 2:
                # 使用平面之间的距离作为高度
                plane1 = BRepAdaptor_Surface(plane_faces[0].face).Plane()
                plane2 = BRepAdaptor_Surface(plane_faces[1].face).Plane()
                height = abs(plane1.Distance(plane2.Location()))
            
            return {
                'type': 'rectangular',
                'width': width,
                'length': length,
                'height': height,
                'location': (top_node.center.X(), top_node.center.Y(), top_node.center.Z())
            }


# 测试代码
if __name__ == "__main__":
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
    
    # 创建一个带孔的盒子
    box = BRepPrimAPI_MakeBox(100, 100, 20).Shape()
    cylinder = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50, 50, 0), gp_Dir(0, 0, 1)), 10, 30).Shape()
    shape_with_hole = BRepAlgoAPI_Cut(box, cylinder).Shape()
    
    # 识别特征
    recognizer = AAGFeatureRecognizer()
    features = recognizer.recognize(shape_with_hole)
    
    # 打印结果
    print("\n识别到的特征:")
    print(f"- 孔: {len(features['holes'])}")
    for i, hole in enumerate(features['holes']):
        print(f"  孔 #{i+1}:")
        for key, value in hole.items():
            print(f"    {key}: {value}")
    
    print(f"- 槽: {len(features['slots'])}")
    for i, slot in enumerate(features['slots']):
        print(f"  槽 #{i+1}:")
        for key, value in slot.items():
            print(f"    {key}: {value}")
    
    print(f"- 凸台: {len(features['bosses'])}")
    for i, boss in enumerate(features['bosses']):
        print(f"  凸台 #{i+1}:")
        for key, value in boss.items():
            print(f"    {key}: {value}")
