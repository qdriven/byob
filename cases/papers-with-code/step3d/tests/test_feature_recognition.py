"""
测试STEP文件特征识别功能
"""

import os
import pytest
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from step3d.core import StepReader
from step3d.features import FeatureRecognizer

@pytest.fixture
def step_file_path():
    """获取测试用的STEP文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "test_files", "test_part.stp")

@pytest.fixture
def loaded_shape(step_file_path):
    """加载STEP文件并返回形状对象"""
    reader = StepReader()
    shape = reader.read_file(step_file_path)
    assert isinstance(shape, TopoDS_Shape), "加载的对象不是TopoDS_Shape类型"
    return shape

@pytest.fixture
def feature_recognizer():
    """创建特征识别器实例"""
    return FeatureRecognizer()

def test_basic_shape_loading(loaded_shape):
    """测试基本的形状加载"""
    assert loaded_shape is not None
    
    # 验证形状中包含面
    explorer = TopExp_Explorer(loaded_shape, TopAbs_FACE)
    face_count = 0
    while explorer.More():
        face_count += 1
        explorer.Next()
    
    assert face_count > 0, "形状中没有找到任何面"
    print(f"形状中包含 {face_count} 个面")

def test_hole_recognition(loaded_shape, feature_recognizer):
    """测试孔特征识别"""
    features = feature_recognizer.recognize(loaded_shape)
    
    # 验证特征字典结构
    assert 'holes' in features, "特征中缺少holes键"
    assert isinstance(features['holes'], list), "holes应该是列表类型"
    
    # 测试孔特征
    holes = features['holes']
    assert len(holes) == 5, f"应该识别出5个孔，但实际识别出 {len(holes)} 个"
    
    print(f"\n识别到 {len(holes)} 个孔特征:")
    for i, hole in enumerate(holes, 1):
        print(f"\n孔 #{i}:")
        print(f"- 类型: {hole['type']}")
        print(f"- 半径: {hole['radius']:.2f}")
        print(f"- 深度: {hole['depth']:.2f}")
        print(f"- 位置: ({hole['location'][0]:.2f}, {hole['location'][1]:.2f}, {hole['location'][2]:.2f})")
        
        # 验证孔的基本参数
        assert 'type' in hole, "孔特征缺少type字段"
        assert 'radius' in hole, "孔特征缺少radius字段"
        assert 'depth' in hole, "孔特征缺少depth字段"
        assert 'location' in hole, "孔特征缺少location字段"
        assert 'axis' in hole, "孔特征缺少axis字段"
        
        # 验证参数的合理性
        assert hole['radius'] > 0, "孔的半径应该大于0"
        assert hole['depth'] > 0, "孔的深度应该大于0"
        assert isinstance(hole['location'], tuple) and len(hole['location']) == 3, "位置应该是三维坐标"
        assert isinstance(hole['axis'], tuple) and len(hole['axis']) == 3, "轴向应该是三维向量"

def test_slot_recognition(loaded_shape, feature_recognizer):
    """测试槽特征识别"""
    features = feature_recognizer.recognize(loaded_shape)
    
    # 验证特征字典结构
    assert 'slots' in features, "特征中缺少slots键"
    assert isinstance(features['slots'], list), "slots应该是列表类型"
    
    # 测试槽特征
    slots = features['slots']
    assert len(slots) == 4, f"应该识别出4个槽，但实际识别出 {len(slots)} 个"
    
    print(f"\n识别到 {len(slots)} 个槽特征:")
    for i, slot in enumerate(slots, 1):
        print(f"\n槽 #{i}:")
        print(f"- 类型: {slot['type']}")
        print(f"- 宽度: {slot['width']:.2f}")
        print(f"- 长度: {slot['length']:.2f}")
        print(f"- 深度: {slot['depth']:.2f}")
        print(f"- 位置: ({slot['location'][0]:.2f}, {slot['location'][1]:.2f}, {slot['location'][2]:.2f})")
        
        # 验证槽的基本参数
        assert 'type' in slot, "槽特征缺少type字段"
        assert 'width' in slot, "槽特征缺少width字段"
        assert 'length' in slot, "槽特征缺少length字段"
        assert 'depth' in slot, "槽特征缺少depth字段"
        assert 'location' in slot, "槽特征缺少location字段"
        
        # 验证参数的合理性
        assert slot['width'] > 0, "槽的宽度应该大于0"
        assert slot['length'] > 0, "槽的长度应该大于0"
        assert slot['depth'] > 0, "槽的深度应该大于0"
        assert isinstance(slot['location'], tuple) and len(slot['location']) == 3, "位置应该是三维坐标"

def test_feature_relationships(loaded_shape, feature_recognizer):
    """测试特征之间的关系"""
    features = feature_recognizer.recognize(loaded_shape)
    holes = features['holes']
    slots = features['slots']
    
    # 检查特征位置的合理性
    hole_locations = [(h['location'][0], h['location'][1]) for h in holes]
    slot_locations = [(s['location'][0], s['location'][1]) for s in slots]
    
    # 验证没有重叠的特征
    all_locations = hole_locations + slot_locations
    unique_locations = set(all_locations)
    assert len(all_locations) == len(unique_locations), "存在重叠的特征"

def test_internal_feature_detection(loaded_shape, feature_recognizer):
    """测试内部特征检测"""
    # 获取一个圆柱面
    explorer = TopExp_Explorer(loaded_shape, TopAbs_FACE)
    cylinder_face = None
    while explorer.More():
        face = topods.Face(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder_face = face
            break
        explorer.Next()
    
    assert cylinder_face is not None, "没有找到圆柱面用于测试"
    
    # 测试内部特征检测
    is_internal = feature_recognizer._is_internal_feature(cylinder_face)
    assert isinstance(is_internal, bool), "_is_internal_feature应该返回布尔值"

def test_face_collection(loaded_shape, feature_recognizer):
    """测试相关面的收集"""
    # 获取一个起始面
    explorer = TopExp_Explorer(loaded_shape, TopAbs_FACE)
    start_face = topods.Face(explorer.Current())
    
    # 测试相关面收集
    related_faces = feature_recognizer._collect_related_faces(loaded_shape, start_face)
    assert isinstance(related_faces, set), "_collect_related_faces应该返回一个集合"
    assert len(related_faces) > 0, "应该至少收集到一个相关面"
    assert start_face in related_faces, "相关面集合应该包含起始面"

def test_geometric_properties(loaded_shape, feature_recognizer):
    """测试几何属性计算"""
    features = feature_recognizer.recognize(loaded_shape)
    
    for hole in features['holes']:
        # 测试轴向向量的规范化
        axis = hole['axis']
        axis_length = (axis[0]**2 + axis[1]**2 + axis[2]**2)**0.5
        assert abs(axis_length - 1.0) < 0.001, "轴向向量应该是单位向量"
    
    for slot in features['slots']:
        # 测试槽的尺寸比例
        assert slot['length'] >= slot['width'], "槽的长度应该大于或等于宽度" 