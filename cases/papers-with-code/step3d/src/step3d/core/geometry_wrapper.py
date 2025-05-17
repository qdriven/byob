"""
几何处理包装类，简化PythonOCC的使用
"""

from typing import List, Dict, Any
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, 
                             GeomAbs_Cone, GeomAbs_Sphere)
from OCCUtils.face import Face
from OCCUtils.edge import Edge
from OCCUtils.surface import Surface

class GeometryWrapper:
    """几何处理包装类"""
    
    @staticmethod
    def get_face_type(face: TopoDS_Face) -> str:
        """获取面的类型"""
        occ_face = Face(face)
        return occ_face.surface_type
    
    @staticmethod
    def get_face_parameters(face: TopoDS_Face) -> Dict[str, Any]:
        """获取面的参数"""
        occ_face = Face(face)
        surface = occ_face.surface
        
        params = {
            "type": occ_face.surface_type
        }
        
        if occ_face.surface_type == "plane":
            params.update({
                "normal": surface.normal,
                "location": surface.location
            })
        elif occ_face.surface_type == "cylinder":
            params.update({
                "radius": surface.radius,
                "axis": surface.axis,
                "location": surface.location
            })
        elif occ_face.surface_type == "cone":
            params.update({
                "radius": surface.radius,
                "semi_angle": surface.semi_angle,
                "axis": surface.axis,
                "location": surface.location
            })
        elif occ_face.surface_type == "sphere":
            params.update({
                "radius": surface.radius,
                "center": surface.location
            })
            
        return params
    
    @staticmethod
    def get_edge_type(edge: TopoDS_Edge) -> str:
        """获取边的类型"""
        occ_edge = Edge(edge)
        return occ_edge.curve_type
    
    @staticmethod
    def get_edge_parameters(edge: TopoDS_Edge) -> Dict[str, Any]:
        """获取边的参数"""
        occ_edge = Edge(edge)
        curve = occ_edge.curve
        
        params = {
            "type": occ_edge.curve_type,
            "start_point": curve.start_point,
            "end_point": curve.end_point
        }
        
        if occ_edge.curve_type == "circle":
            params.update({
                "radius": curve.radius,
                "center": curve.center,
                "axis": curve.axis
            })
        elif occ_edge.curve_type == "ellipse":
            params.update({
                "major_radius": curve.major_radius,
                "minor_radius": curve.minor_radius,
                "center": curve.center,
                "axis": curve.axis
            })
            
        return params 