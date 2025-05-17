"""
几何数据转换工具，用于在OpenCASCADE和自定义数据结构之间进行转换
"""

from typing import List, Dict, Any
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                             GeomAbs_Hyperbola, GeomAbs_Parabola)
from OCC.Core.gp import gp_Pnt

from .geometry import (Shape, Face, Edge, Point, Vector, Surface,
                      Plane, Cylinder, Cone, Sphere)

class GeometryConverter:
    """几何数据转换器"""
    
    @staticmethod
    def extract_vertices(shape: TopoDS_Shape) -> List[Point]:
        """提取形状中的所有顶点"""
        vertices = []
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            pnt = vertex.Pnt()
            vertices.append(Point.from_gp_pnt(pnt))
            explorer.Next()
        return vertices
    
    @staticmethod
    def extract_edges(shape: TopoDS_Shape) -> List[Edge]:
        """提取形状中的所有边"""
        edges = []
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            curve = BRepAdaptor_Curve(edge)
            
            # 获取起点和终点
            start_pnt = curve.Value(curve.FirstParameter())
            end_pnt = curve.Value(curve.LastParameter())
            
            # 确定曲线类型
            curve_type = GeometryConverter._get_curve_type_name(curve.GetType())
            
            # 提取曲线参数
            parameters = GeometryConverter._extract_curve_parameters(curve)
            
            edges.append(Edge(
                start_point=Point.from_gp_pnt(start_pnt),
                end_point=Point.from_gp_pnt(end_pnt),
                curve_type=curve_type,
                parameters=parameters
            ))
            explorer.Next()
        return edges
    
    @staticmethod
    def extract_faces(shape: TopoDS_Shape) -> List[Face]:
        """提取形状中的所有面"""
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            try:
                faces.append(Face.from_topods_face(face))
            except ValueError as e:
                print(f"警告：无法处理面 - {str(e)}")
            explorer.Next()
        return faces
    
    @staticmethod
    def _get_curve_type_name(curve_type: int) -> str:
        """获取曲线类型的名称"""
        type_map = {
            GeomAbs_Line: "line",
            GeomAbs_Circle: "circle",
            GeomAbs_Ellipse: "ellipse",
            GeomAbs_Hyperbola: "hyperbola",
            GeomAbs_Parabola: "parabola"
        }
        return type_map.get(curve_type, "unknown")
    
    @staticmethod
    def _extract_curve_parameters(curve: BRepAdaptor_Curve) -> Dict[str, Any]:
        """提取曲线的参数"""
        params = {}
        curve_type = curve.GetType()
        
        if curve_type == GeomAbs_Circle:
            circle = curve.Circle()
            params.update({
                "radius": circle.Radius(),
                "center": Point.from_gp_pnt(circle.Location()).to_dict(),
                "axis": Vector.from_gp_vec(circle.Axis().Direction()).to_dict()
            })
        elif curve_type == GeomAbs_Ellipse:
            ellipse = curve.Ellipse()
            params.update({
                "major_radius": ellipse.MajorRadius(),
                "minor_radius": ellipse.MinorRadius(),
                "center": Point.from_gp_pnt(ellipse.Location()).to_dict(),
                "axis": Vector.from_gp_vec(ellipse.Axis().Direction()).to_dict()
            })
        
        return params
    
    @classmethod
    def convert_shape(cls, shape: TopoDS_Shape) -> Shape:
        """将OpenCASCADE形状转换为Shape对象"""
        vertices = cls.extract_vertices(shape)
        edges = cls.extract_edges(shape)
        faces = cls.extract_faces(shape)
        
        return Shape(
            faces=faces,
            edges=edges,
            vertices=vertices
        ) 