"""
STEP文件分析示例
"""

import sys
from pathlib import Path

from step3d.core.step_reader import StepReader
from step3d.core.geometry_converter import GeometryConverter

def analyze_step_file(file_path: str):
    """分析STEP文件中的几何特征"""
    # 读取STEP文件
    reader = StepReader()
    shape = reader.read_file(file_path)
    
    # 获取面的类型统计
    surface_types = shape.get_surface_types()
    
    # 打印分析结果
    print("\n几何特征分析结果:")
    print("-" * 50)
    print("面类型统计:")
    for surface_type, count in surface_types.items():
        print(f"  - {surface_type}: {count}个")
    
    print("\n详细几何信息:")
    print("-" * 50)
    for i, face in enumerate(shape.faces, 1):
        print(f"\n面 {i}:")
        print(f"  类型: {face.surface.type}")
        if isinstance(face.surface, Plane):
            print(f"  法向量: {face.surface.normal.to_dict()}")
        elif isinstance(face.surface, Cylinder):
            print(f"  半径: {face.surface.radius}")
            print(f"  轴线: {face.surface.axis.to_dict()}")
        elif isinstance(face.surface, Cone):
            print(f"  半径: {face.surface.radius}")
            print(f"  半角: {face.surface.semi_angle}")
            print(f"  轴线: {face.surface.axis.to_dict()}")
        elif isinstance(face.surface, Sphere):
            print(f"  半径: {face.surface.radius}")
            print(f"  中心: {face.surface.center.to_dict()}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python analyze_step.py <step_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    
    try:
        analyze_step_file(file_path)
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 