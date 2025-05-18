"""
VTK可视化工具函数
"""

import vtk
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_FACE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Poly import Poly_Triangulation

def create_vtk_actor_from_shape(shape: TopoDS_Shape) -> vtk.vtkActor:
    """
    从OpenCASCADE形状创建VTK Actor
    
    Args:
        shape: OpenCASCADE形状对象
        
    Returns:
        vtk.vtkActor: VTK渲染用的actor对象
    """
    # 创建网格
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    
    # 创建VTK数据结构
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    
    # 遍历所有面
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is not None:
            # 添加顶点
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i)
                points.InsertNextPoint(point.X(), point.Y(), point.Z())
            
            # 添加三角形
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(3)
                for j in range(3):
                    polygon.GetPointIds().SetId(j, triangle.Value(j+1) - 1)
                cells.InsertNextCell(polygon)
        
        explorer.Next()
    
    # 创建PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    
    # 创建Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # 创建Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 设置显示属性
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 灰色
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetAmbient(0.1)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.2)
    
    return actor 