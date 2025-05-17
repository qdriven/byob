"""
STEP文件读取和基本几何提取
"""

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape
from .geometry import Shape
from .geometry_converter import GeometryConverter

class StepReader:
    """STEP文件读取和处理类"""
    
    def __init__(self):
        self.reader = STEPControl_Reader()
        
    def read_file(self, filename: str) -> Shape:
        """
        读取STEP文件并返回形状对象
        
        Args:
            filename: STEP文件路径
            
        Returns:
            Shape: 包含几何数据的形状对象
            
        Raises:
            Exception: 如果文件读取失败
        """
        status = self.reader.ReadFile(filename)
        
        if status == IFSelect_RetDone:
            self.reader.TransferRoot()
            topods_shape = self.reader.Shape()
            return GeometryConverter.convert_shape(topods_shape)
        else:
            raise Exception("错误：STEP文件读取失败")