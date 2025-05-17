"""
几何数据结构模块，用于表示STEP文件中的基本几何元素
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, 
                             GeomAbs_Cone, GeomAbs_Sphere)

from .geometry_wrapper import GeometryWrapper

@dataclass
class Point:
    """3D点数据结构"""
    x: float
    y: float
    z: float
    
    @classmethod
    def from_gp_pnt(cls, pnt: gp_Pnt) -> 'Point':
        """从OpenCASCADE的gp_Pnt创建Point对象"""
        return cls(pnt.X(), pnt.Y(), pnt.Z())
    
    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z
        }

@dataclass
class Vector:
    """3D向量数据结构"""
    x: float
    y: float
    z: float
    
    @classmethod
    def from_gp_vec(cls, vec: gp_Vec) -> 'Vector':
        """从OpenCASCADE的gp_Vec创建Vector对象"""
        return cls(vec.X(), vec.Y(), vec.Z())
    
    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z
        }

@dataclass
class Surface:
    """曲面基类"""
    type: str
    parameters: dict
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type,
            "parameters": self.parameters
        }

@dataclass
class Plane(Surface):
    """平面数据结构"""
    origin: Point
    normal: Vector
    
    @classmethod
    def from_face(cls, face: TopoDS_Face) -> 'Plane':
        """从OpenCASCADE的面创建Plane对象"""
        params = GeometryWrapper.get_face_parameters(face)
        if params["type"] != "plane":
            raise ValueError("面不是平面")
        
        return cls(
            type="plane",
            parameters=params,
            origin=Point.from_gp_pnt(params["location"]),
            normal=Vector.from_gp_vec(params["normal"])
        )

@dataclass
class Cylinder(Surface):
    """圆柱面数据结构"""
    origin: Point
    axis: Vector
    radius: float
    
    @classmethod
    def from_face(cls, face: TopoDS_Face) -> 'Cylinder':
        """从OpenCASCADE的面创建Cylinder对象"""
        params = GeometryWrapper.get_face_parameters(face)
        if params["type"] != "cylinder":
            raise ValueError("面不是圆柱面")
        
        return cls(
            type="cylinder",
            parameters=params,
            origin=Point.from_gp_pnt(params["location"]),
            axis=Vector.from_gp_vec(params["axis"]),
            radius=params["radius"]
        )

@dataclass
class Cone(Surface):
    """圆锥面数据结构"""
    origin: Point
    axis: Vector
    radius: float
    semi_angle: float
    
    @classmethod
    def from_face(cls, face: TopoDS_Face) -> 'Cone':
        """从OpenCASCADE的面创建Cone对象"""
        params = GeometryWrapper.get_face_parameters(face)
        if params["type"] != "cone":
            raise ValueError("面不是圆锥面")
        
        return cls(
            type="cone",
            parameters=params,
            origin=Point.from_gp_pnt(params["location"]),
            axis=Vector.from_gp_vec(params["axis"]),
            radius=params["radius"],
            semi_angle=params["semi_angle"]
        )

@dataclass
class Sphere(Surface):
    """球面数据结构"""
    center: Point
    radius: float
    
    @classmethod
    def from_face(cls, face: TopoDS_Face) -> 'Sphere':
        """从OpenCASCADE的面创建Sphere对象"""
        params = GeometryWrapper.get_face_parameters(face)
        if params["type"] != "sphere":
            raise ValueError("面不是球面")
        
        return cls(
            type="sphere",
            parameters=params,
            center=Point.from_gp_pnt(params["center"]),
            radius=params["radius"]
        )

@dataclass
class Edge:
    """边数据结构"""
    start_point: Point
    end_point: Point
    curve_type: str
    parameters: dict
    
    @classmethod
    def from_topods_edge(cls, edge: TopoDS_Edge) -> 'Edge':
        """从OpenCASCADE的边创建Edge对象"""
        params = GeometryWrapper.get_edge_parameters(edge)
        
        return cls(
            start_point=Point.from_gp_pnt(params["start_point"]),
            end_point=Point.from_gp_pnt(params["end_point"]),
            curve_type=params["type"],
            parameters=params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "curve_type": self.curve_type,
            "start_point": self.start_point.to_dict(),
            "end_point": self.end_point.to_dict(),
            "parameters": self.parameters
        }

@dataclass
class Face:
    """面数据结构"""
    surface: Surface
    edges: List[Edge]
    vertices: List[Point]
    
    @classmethod
    def from_topods_face(cls, face: TopoDS_Face) -> 'Face':
        """从OpenCASCADE的面创建Face对象"""
        surface_type = GeometryWrapper.get_face_type(face)
        
        # 根据面类型创建相应的Surface对象
        if surface_type == "plane":
            surface_obj = Plane.from_face(face)
        elif surface_type == "cylinder":
            surface_obj = Cylinder.from_face(face)
        elif surface_type == "cone":
            surface_obj = Cone.from_face(face)
        elif surface_type == "sphere":
            surface_obj = Sphere.from_face(face)
        else:
            raise ValueError(f"不支持的面类型: {surface_type}")
        
        # TODO: 提取边和顶点信息
        edges = []
        vertices = []
        
        return cls(
            surface=surface_obj,
            edges=edges,
            vertices=vertices
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "surface": self.surface.to_dict(),
            "edges": [edge.to_dict() for edge in self.edges],
            "vertices": [vertex.to_dict() for vertex in self.vertices]
        }

@dataclass
class Shape:
    """形状数据结构，表示完整的STEP模型"""
    faces: List[Face]
    edges: List[Edge]
    vertices: List[Point]
    
    @classmethod
    def from_topods_shape(cls, shape: TopoDS_Shape) -> 'Shape':
        """从OpenCASCADE的形状创建Shape对象"""
        # TODO: 实现从TopoDS_Shape到Shape的转换
        faces = []
        edges = []
        vertices = []
        
        return cls(
            faces=faces,
            edges=edges,
            vertices=vertices
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "faces": [face.to_dict() for face in self.faces],
            "edges": [edge.to_dict() for edge in self.edges],
            "vertices": [vertex.to_dict() for vertex in self.vertices]
        }
    
    def get_surface_types(self) -> Dict[str, int]:
        """获取所有面的类型统计"""
        surface_types = {}
        for face in self.faces:
            surface_type = face.surface.type
            surface_types[surface_type] = surface_types.get(surface_type, 0) + 1
        return surface_types 