import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
from data_process import (
    read_observation_data,
    interpolate_to_grid,
    safe_int_conversion,
    is_numeric_string,
    read_model_data
)


class TestReadObservationData:
    @pytest.fixture
    def sample_observation_file(self):
        """创建一个临时的观测数据文件用于测试"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b"Year,Mon,Day,Hour,Lat,Lon,PRE_1h\n")
            f.write(b"2021,7,1,0,35.5,112.5,1.5\n")
            f.write(b"2021,7,1,1,35.5,112.5,2.0\n")
            f.write(b"2021,7,1,2,35.5,112.5,0.0\n")
            f.write(b"2021,7,2,0,35.5,112.5,3.0\n")
            f.write(b"2021,7,2,1,35.5,112.5,1.0\n")
            f.write(b"2021,6,30,0,35.5,112.5,5.0\n")  # 不在目标月份
            f.write(b"2020,7,1,0,35.5,112.5,4.0\n")   # 不在目标年份
        return f.name

    def test_read_observation_data(self, sample_observation_file):
        """测试读取观测数据函数"""
        result = read_observation_data(sample_observation_file, targe_year=2021, target_month=7)
        
        # 检查返回的DataFrame是否符合预期
        assert isinstance(result, pd.DataFrame)
        assert 'lon' in result.columns
        assert 'lat' in result.columns
        assert 'rainfall' in result.columns
        assert 'date' in result.columns
        
        # 检查数据筛选是否正确（只有2021年7月的数据）
        assert len(result) == 2  # 应该有两个日期（7月1日和7月2日）
        
        # 检查日降水量计算是否正确
        # 7月1日的降水量应该是1.5 + 2.0 + 0.0 = 3.5
        # 7月2日的降水量应该是3.0 + 1.0 = 4.0
        date_1 = pd.to_datetime('2021-07-01')
        date_2 = pd.to_datetime('2021-07-02')
        
        rainfall_1 = result[result['date'] == date_1]['rainfall'].values[0]
        rainfall_2 = result[result['date'] == date_2]['rainfall'].values[0]
        
        assert rainfall_1 == 3.5
        assert rainfall_2 == 4.0
        
        # 清理临时文件
        os.unlink(sample_observation_file)


class TestInterpolateToGrid:
    @pytest.fixture
    def sample_daily_rainfall(self):
        """创建一个示例的日降水量DataFrame用于测试"""
        data = {
            'lon': [110.5, 111.5, 112.5, 113.5],
            'lat': [32.5, 33.5, 34.5, 35.5],
            'rainfall': [10.0, 20.0, 30.0, 40.0],
            'date': [
                pd.to_datetime('2021-07-01'),
                pd.to_datetime('2021-07-01'),
                pd.to_datetime('2021-07-01'),
                pd.to_datetime('2021-07-01')
            ]
        }
        return pd.DataFrame(data)

    def test_interpolate_to_grid(self, sample_daily_rainfall):
        """测试站点数据插值到规则网格函数"""
        result = interpolate_to_grid(
            sample_daily_rainfall,
            lon_min=110,
            lon_max=114,
            lat_min=32,
            lat_max=36,
            resolution=0.5
        )
        
        # 检查返回的字典是否符合预期
        assert isinstance(result, dict)
        assert len(result) == 1  # 只有一个日期
        
        date = pd.to_datetime('2021-07-01')
        assert date in result
        
        # 检查网格数据的结构
        grid_data = result[date]
        assert 'lon' in grid_data
        assert 'lat' in grid_data
        assert 'rainfall' in grid_data
        
        # 检查网格尺寸
        assert len(grid_data['lon']) == 9  # (114-110)/0.5 + 1 = 9
        assert len(grid_data['lat']) == 9  # (36-32)/0.5 + 1 = 9
        assert grid_data['rainfall'].shape == (9, 9)
        
        # 检查插值结果在已知点的值是否接近原始值
        # 注意：由于插值的性质，可能不会完全相等
        lon_idx_110_5 = np.where(np.isclose(grid_data['lon'], 110.5))[0][0]
        lat_idx_32_5 = np.where(np.isclose(grid_data['lat'], 32.5))[0][0]
        assert np.isclose(grid_data['rainfall'][lat_idx_32_5, lon_idx_110_5], 10.0, atol=1.0)
        
        lon_idx_113_5 = np.where(np.isclose(grid_data['lon'], 113.5))[0][0]
        lat_idx_35_5 = np.where(np.isclose(grid_data['lat'], 35.5))[0][0]
        assert np.isclose(grid_data['rainfall'][lat_idx_35_5, lon_idx_113_5], 40.0, atol=1.0)


class TestSafeIntConversion:
    @pytest.mark.parametrize("value,expected", [
        (5, 5),
        (5.7, 5),
        ("5", 5),
        ("5.7", 5),
        ("invalid", 0),
        (None, 0),
    ])
    def test_safe_int_conversion(self, value, expected):
        """测试安全整数转换函数"""
        result = safe_int_conversion(value)
        assert result == expected
    
    def test_safe_int_conversion_with_custom_default(self):
        """测试带有自定义默认值的安全整数转换"""
        result = safe_int_conversion("invalid", default=999)
        assert result == 999


class TestIsNumericString:
    @pytest.mark.parametrize("value,expected", [
        ("5", True),
        ("5.7", True),
        ("-5.7", True),
        ("1e3", True),
        ("invalid", False),
        ("", False),
        (None, False),
    ])
    def test_is_numeric_string(self, value, expected):
        """测试是否为数值字符串函数"""
        result = is_numeric_string(value)
        assert result == expected


class TestReadModelData:
    @pytest.fixture
    def sample_model_data_folder(self):
        """创建一个临时文件夹，包含示例模式数据文件用于测试"""
        with tempfile.TemporaryDirectory() as tmpdirname:
            # 创建一个简化格式的模式数据文件
            with open(os.path.join(tmpdirname, '210701.txt'), 'w') as f:
                f.write("0.125 -0.125 105.000 125.000 40.000 25.000 161 121 10.000 0.000 100.000 1.000 0.000\n")
                # 写入一些示例数据（简化为一个小网格）
                for _ in range(161 * 121):
                    f.write("5.0\n")
            
            # 创建一个完整格式的模式数据文件
            with open(os.path.join(tmpdirname, '210702.txt'), 'w') as f:
                f.write("diamond 4 21070220.036__ECMWF_HR_RAIN24(units:mm) 21 7 2 20 36 9999 0.125 -0.125 105.000 125.000 40.000 25.000 161 121 10.000 0.000 100.000 1.000 0.000\n")
                # 写入一些示例数据（简化为一个小网格）
                for _ in range(161 * 121):
                    f.write("10.0\n")
            
            # 创建一个格式错误的模式数据文件
            with open(os.path.join(tmpdirname, '210703.txt'), 'w') as f:
                f.write("invalid header\n")
                f.write("5.0\n")
            
            return tmpdirname
    
    def test_read_model_data(self, sample_model_data_folder):
        """测试读取模式数据函数"""
        result = read_model_data(sample_model_data_folder)
        
        # 检查返回的字典是否符合预期
        assert isinstance(result, dict)
        
        # 应该成功读取两个文件（210701.txt和210702.txt）
        assert len(result) >= 1
        
        # 检查日期是否正确
        date_1 = pd.to_datetime('2021-07-01')
        if date_1 in result:
            # 检查数据结构
            data_1 = result[date_1]
            assert 'lon' in data_1
            assert 'lat' in data_1
            assert 'rainfall' in data_1
            
            # 检查数据是否在研究区域内
            assert np.all(data_1['lon'] >= 110)
            assert np.all(data_1['lon'] <= 114)
            assert np.all(data_1['lat'] >= 32)
            assert np.all(data_1['lat'] <= 36)
            
            # 检查降水值是否正确
            assert np.all(data_1['rainfall'] == 5.0)
        
        date_2 = pd.to_datetime('2021-07-02')
        if date_2 in result:
            # 检查数据结构
            data_2 = result[date_2]
            assert 'lon' in data_2
            assert 'lat' in data_2
            assert 'rainfall' in data_2
            
            # 检查降水值是否正确
            assert np.all(data_2['rainfall'] == 10.0)


if __name__ == "__main__":
    pytest.main(["-v"])