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
    """获取测试用STEP文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    return os.path.join(root_dir, "docs", "Part_Smiely_holes_only.STEP")

@pytest.fixture
def step_reader():
    """创建StepReader实例"""
    return StepReader()

@pytest.fixture
def feature_recognizer():
    """创建FeatureRecognizer实例"""
    return FeatureRecognizer()

@pytest.fixture
def loaded_shape(step_reader, step_file_path) -> TopoDS_Shape:
    """加载STEP文件并返回形状"""
    return step_reader.read_file(step_file_path)

def test_step_file_exists(step_file_path):
    """测试STEP文件是否存在"""
    assert os.path.exists(step_file_path), f"STEP文件不存在: {step_file_path}"

def test_step_file_loading(loaded_shape):
    """测试STEP文件加载"""
    assert loaded_shape is not None, "STEP文件加载失败"
    assert isinstance(loaded_shape, TopoDS_Shape), "加载的对象不是TopoDS_Shape类型"

def count_faces_by_type(shape: TopoDS_Shape):
    """统计不同类型面的数量"""
    cylinder_count = 0
    plane_count = 0
    
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        surface_type = surface.GetType()
        
        if surface_type == GeomAbs_Cylinder:
            cylinder_count += 1
        elif surface_type == GeomAbs_Plane:
            plane_count += 1
            
        explorer.Next()
        
    return cylinder_count, plane_count

def test_collect_hole_faces(loaded_shape, feature_recognizer):
    """测试孔特征面的收集"""
    # 获取所有可能的孔面集合
    hole_face_sets = feature_recognizer._collect_potential_hole_faces(loaded_shape)
    
    # 验证是否找到了面集合
    assert len(hole_face_sets) > 0, "没有找到任何孔特征面集合"
    
    # 验证每个面集合
    for face_set in hole_face_sets:
        # 每个孔至少应该有一个圆柱面
        has_cylinder = False
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Cylinder:
                has_cylinder = True
                break
        assert has_cylinder, "孔特征面集合中没有圆柱面"
        
        # 打印面集合信息
        cylinder_count, plane_count = 0, 0
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder_count += 1
            elif surface.GetType() == GeomAbs_Plane:
                plane_count += 1
        
        print(f"\n孔特征面集合:")
        print(f"- 圆柱面数量: {cylinder_count}")
        print(f"- 平面数量: {plane_count}")
        print(f"- 总面数: {len(face_set)}")

def test_collect_slot_faces(loaded_shape, feature_recognizer):
    """测试槽特征面的收集"""
    # 获取所有可能的槽面集合
    slot_face_sets = feature_recognizer._collect_potential_slot_faces(loaded_shape)
    
    # 验证是否找到了面集合
    assert len(slot_face_sets) > 0, "没有找到任何槽特征面集合"
    
    # 验证每个面集合
    for face_set in slot_face_sets:
        # 每个槽至少应该有一个平面（底面）
        has_plane = False
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Plane:
                has_plane = True
                break
        assert has_plane, "槽特征面集合中没有平面"
        
        # 打印面集合信息
        cylinder_count, plane_count = 0, 0
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder_count += 1
            elif surface.GetType() == GeomAbs_Plane:
                plane_count += 1
        
        print(f"\n槽特征面集合:")
        print(f"- 圆柱面数量: {cylinder_count}")
        print(f"- 平面数量: {plane_count}")
        print(f"- 总面数: {len(face_set)}")

def test_collect_boss_faces(loaded_shape, feature_recognizer):
    """测试凸台特征面的收集"""
    # 获取所有可能的凸台面集合
    boss_face_sets = feature_recognizer._collect_potential_boss_faces(loaded_shape)
    
    # 验证是否找到了面集合
    assert len(boss_face_sets) > 0, "没有找到任何凸台特征面集合"
    
    # 验证每个面集合
    for face_set in boss_face_sets:
        # 每个凸台应该至少有一个顶面（平面）或侧面（圆柱面）
        has_valid_face = False
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() in [GeomAbs_Cylinder, GeomAbs_Plane]:
                has_valid_face = True
                break
        assert has_valid_face, "凸台特征面集合中没有有效面"
        
        # 打印面集合信息
        cylinder_count, plane_count = 0, 0
        for face in face_set:
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Cylinder:
                cylinder_count += 1
            elif surface.GetType() == GeomAbs_Plane:
                plane_count += 1
        
        print(f"\n凸台特征面集合:")
        print(f"- 圆柱面数量: {cylinder_count}")
        print(f"- 平面数量: {plane_count}")
        print(f"- 总面数: {len(face_set)}")

def test_model_statistics(loaded_shape):
    """测试模型的基本统计信息"""
    # 统计整个模型中的面类型
    cylinder_count, plane_count = count_faces_by_type(loaded_shape)
    
    print(f"\n模型统计信息:")
    print(f"- 圆柱面总数: {cylinder_count}")
    print(f"- 平面总数: {plane_count}")
    
    # 基本验证
    assert cylinder_count > 0, "模型中没有圆柱面"
    assert plane_count > 0, "模型中没有平面"

def test_feature_recognition(loaded_shape, feature_recognizer):
    """测试特征识别功能"""
    features = feature_recognizer.recognize(loaded_shape)
    
    # 验证特征字典结构
    expected_features = ['holes', 'slots', 'bosses']
    for feature_type in expected_features:
        assert feature_type in features, f"特征中缺少{feature_type}键"
        assert isinstance(features[feature_type], list), f"{feature_type}应该是列表类型"
    
    # 测试孔特征
    holes = features['holes']
    if len(holes) > 0:
        print(f"\n识别到 {len(holes)} 个孔特征:")
        for i, hole in enumerate(holes, 1):
            print(f"\n孔 #{i}:")
            print(f"- 类型: {hole['type']}")
            print(f"- 半径: {hole['radius']:.2f}")
            print(f"- 位置: {hole['location']}")
            print(f"- 轴向: {hole['axis']}")
            
            assert 'type' in hole, "孔特征缺少type属性"
            assert 'radius' in hole, "孔特征缺少radius属性"
            assert 'location' in hole, "孔特征缺少location属性"
            assert 'axis' in hole, "孔特征缺少axis属性"
            
            assert isinstance(hole['radius'], float), "radius应该是浮点数类型"
            assert isinstance(hole['location'], tuple), "location应该是元组类型"
            assert isinstance(hole['axis'], tuple), "axis应该是元组类型"
            assert len(hole['location']) == 3, "location应该包含3个坐标值"
            assert len(hole['axis']) == 3, "axis应该包含3个方向值"
    
    # 测试槽特征
    slots = features['slots']
    if len(slots) > 0:
        print(f"\n识别到 {len(slots)} 个槽特征:")
        for i, slot in enumerate(slots, 1):
            print(f"\n槽 #{i}:")
            print(f"- 类型: {slot['type']}")
            print(f"- 宽度: {slot.get('width', 'N/A')}")
            print(f"- 深度: {slot.get('depth', 'N/A')}")
            print(f"- 位置: {slot.get('location', 'N/A')}")
            print(f"- 方向: {slot.get('direction', 'N/A')}")
            
            assert 'type' in slot, "槽特征缺少type属性"
            if 'width' in slot:
                assert isinstance(slot['width'], float), "width应该是浮点数类型"
            if 'depth' in slot:
                assert isinstance(slot['depth'], float), "depth应该是浮点数类型"
            if 'location' in slot:
                assert isinstance(slot['location'], tuple), "location应该是元组类型"
                assert len(slot['location']) == 3, "location应该包含3个坐标值"
    
    # 测试凸台特征
    bosses = features['bosses']
    if len(bosses) > 0:
        print(f"\n识别到 {len(bosses)} 个凸台特征:")
        for i, boss in enumerate(bosses, 1):
            print(f"\n凸台 #{i}:")
            print(f"- 类型: {boss['type']}")
            print(f"- 高度: {boss.get('height', 'N/A')}")
            print(f"- 直径: {boss.get('diameter', 'N/A')}")
            print(f"- 位置: {boss.get('location', 'N/A')}")
            
            assert 'type' in boss, "凸台特征缺少type属性"
            if 'height' in boss:
                assert isinstance(boss['height'], float), "height应该是浮点数类型"
            if 'diameter' in boss:
                assert isinstance(boss['diameter'], float), "diameter应该是浮点数类型"
            if 'location' in boss:
                assert isinstance(boss['location'], tuple), "location应该是元组类型"
                assert len(boss['location']) == 3, "location应该包含3个坐标值"

def test_hole_positions(loaded_shape, feature_recognizer):
    """测试孔的相对位置关系"""
    features = feature_recognizer.recognize(loaded_shape)
    holes = features['holes']
    
    # 按Y坐标排序孔（从上到下）
    sorted_holes = sorted(holes, key=lambda h: h['location'][1], reverse=True)
    
    if len(sorted_holes) >= 2:
        # 验证上方的孔（眼睛）应该大致在同一水平线上
        if len(sorted_holes) >= 2:
            eye1_y = sorted_holes[0]['location'][1]
            eye2_y = sorted_holes[1]['location'][1]
            assert abs(eye1_y - eye2_y) < 1.0, "两个眼睛的Y坐标差异过大"
            
            print("\n眼睛位置验证:")
            print(f"左眼Y坐标: {eye1_y:.2f}")
            print(f"右眼Y坐标: {eye2_y:.2f}")
            print(f"Y坐标差异: {abs(eye1_y - eye2_y):.2f}")

def test_hole_sizes(loaded_shape, feature_recognizer):
    """测试孔的尺寸"""
    features = feature_recognizer.recognize(loaded_shape)
    holes = features['holes']
    
    # 收集所有孔的半径
    radii = [hole['radius'] for hole in holes]
    
    if radii:
        min_radius = min(radii)
        max_radius = max(radii)
        avg_radius = sum(radii) / len(radii)
        
        print("\n孔尺寸统计:")
        print(f"最小半径: {min_radius:.2f}")
        print(f"最大半径: {max_radius:.2f}")
        print(f"平均半径: {avg_radius:.2f}")
        
        # 验证半径在合理范围内
        assert min_radius > 0, "存在半径小于等于0的孔"
        assert max_radius < 100, "存在半径过大的孔"  # 根据实际模型调整阈值 