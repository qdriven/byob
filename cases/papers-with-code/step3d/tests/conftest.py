"""
Pytest配置文件
"""

import pytest

def pytest_configure(config):
    """配置pytest运行环境"""
    # 添加标记
    config.addinivalue_line(
        "markers",
        "feature: 特征识别相关的测试"
    )

@pytest.fixture(autouse=True)
def run_around_tests():
    """在每个测试前后运行的操作"""
    # 测试开始前的设置
    yield
    # 测试结束后的清理 