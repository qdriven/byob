"""
STEP文件读取模块
"""

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape

class StepReader:
    """STEP文件读取类"""
    
    def __init__(self):
        """初始化STEP读取器"""
        self.reader = STEPControl_Reader()
        
    def read_file(self, filename: str) -> TopoDS_Shape:
        """
        读取STEP文件并返回OpenCASCADE形状对象
        
        Args:
            filename: STEP文件路径
            
        Returns:
            TopoDS_Shape: OpenCASCADE形状对象
            
        Raises:
            Exception: 如果文件读取失败
        """
        status = self.reader.ReadFile(filename)
        
        if status == IFSelect_RetDone:
            self.reader.TransferRoot()
            return self.reader.Shape()
        else:
            raise Exception("错误：STEP文件读取失败")