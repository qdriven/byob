"""
使用Pydantic的几何数据模型
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import numpy as np
from OCC.Core.gp import gp_Pnt, gp_Vec

class Point(BaseModel):
    """3D点数据模型"""
    x: float = Field(..., description="X坐标")
    y: float = Field(..., description="Y坐标")
    z: float = Field(..., description="Z坐标")
    
    @classmethod
    def from_gp_pnt(cls, pnt: gp_Pnt) -> 'Point':
        """从OpenCASCADE的gp_Pnt创建Point对象"""
        return cls(x=pnt.X(), y=pnt.Y(), z=pnt.Z())
    
    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    @validator('x', 'y', 'z')
    def validate_coordinates(cls, v):
        """验证坐标值"""
        if not isinstance(v, (int, float)):
            raise ValueError('坐标必须是数值类型')
        return float(v)

class Vector(BaseModel):
    """3D向量数据模型"""
    x: float = Field(..., description="X分量")
    y: float = Field(..., description="Y分量")
    z: float = Field(..., description="Z分量")
    
    @classmethod
    def from_gp_vec(cls, vec: gp_Vec) -> 'Vector':
        """从OpenCASCADE的gp_Vec创建Vector对象"""
        return cls(x=vec.X(), y=vec.Y(), z=vec.Z())
    
    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    @validator('x', 'y', 'z')
    def validate_components(cls, v):
        """验证向量分量"""
        if not isinstance(v, (int, float)):
            raise ValueError('向量分量必须是数值类型')
        return float(v)

class Surface(BaseModel):
    """曲面基类"""
    type: str = Field(..., description="曲面类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="曲面参数")

class Plane(Surface):
    """平面数据模型"""
    origin: Point = Field(..., description="平面原点")
    normal: Vector = Field(..., description="平面法向量")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "plane",
                "parameters": {},
                "origin": {"x": 0, "y": 0, "z": 0},
                "normal": {"x": 0, "y": 0, "z": 1}
            }
        }

class Cylinder(Surface):
    """圆柱面数据模型"""
    origin: Point = Field(..., description="圆柱原点")
    axis: Vector = Field(..., description="圆柱轴线")
    radius: float = Field(..., gt=0, description="圆柱半径")
    
    @validator('radius')
    def validate_radius(cls, v):
        """验证半径"""
        if v <= 0:
            raise ValueError('半径必须大于0')
        return v

class Cone(Surface):
    """圆锥面数据模型"""
    origin: Point = Field(..., description="圆锥原点")
    axis: Vector = Field(..., description="圆锥轴线")
    radius: float = Field(..., gt=0, description="圆锥半径")
    semi_angle: float = Field(..., gt=0, lt=90, description="圆锥半角(度)")
    
    @validator('radius')
    def validate_radius(cls, v):
        """验证半径"""
        if v <= 0:
            raise ValueError('半径必须大于0')
        return v
    
    @validator('semi_angle')
    def validate_semi_angle(cls, v):
        """验证半角"""
        if not 0 < v < 90:
            raise ValueError('半角必须在0到90度之间')
        return v

class Sphere(Surface):
    """球面数据模型"""
    center: Point = Field(..., description="球心")
    radius: float = Field(..., gt=0, description="球半径")
    
    @validator('radius')
    def validate_radius(cls, v):
        """验证半径"""
        if v <= 0:
            raise ValueError('半径必须大于0')
        return v

class Edge(BaseModel):
    """边数据模型"""
    start_point: Point = Field(..., description="起点")
    end_point: Point = Field(..., description="终点")
    curve_type: str = Field(..., description="曲线类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="曲线参数")

class Face(BaseModel):
    """面数据模型"""
    surface: Surface = Field(..., description="曲面")
    edges: List[Edge] = Field(default_factory=list, description="边列表")
    vertices: List[Point] = Field(default_factory=list, description="顶点列表")

class Shape(BaseModel):
    """形状数据模型"""
    faces: List[Face] = Field(default_factory=list, description="面列表")
    edges: List[Edge] = Field(default_factory=list, description="边列表")
    vertices: List[Point] = Field(default_factory=list, description="顶点列表")
    
    def get_surface_types(self) -> Dict[str, int]:
        """获取所有面的类型统计"""
        surface_types = {}
        for face in self.faces:
            surface_type = face.surface.type
            surface_types[surface_type] = surface_types.get(surface_type, 0) + 1
        return surface_types 