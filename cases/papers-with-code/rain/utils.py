
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import numpy as np
from matplotlib.font_manager import FontProperties

def setup_font():
    """
    配置matplotlib以支持包括中文在内的多语言文本显示。
    使用更可靠的方法确保中文正确显示。
    """
    # 设置高DPI以获得更好的文本渲染效果
    plt.rcParams['figure.dpi'] = 300

    # 确保负号正确显示
    plt.rcParams['axes.unicode_minus'] = False

    # 设置白色背景
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    # 尝试使用通用的字体方法
    # 首先尝试可能支持中文的常见无衬线字体
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans',
        'Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK', 'sans-serif'
    ]

    # 将后备字体族设置为无衬线字体
    plt.rcParams['font.family'] = 'sans-serif'

    # 检测操作系统类型，为不同系统设置最适合的字体
    if sys.platform.startswith('win'):
        # Windows系统优先使用微软雅黑和黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] + plt.rcParams['font.sans-serif']
    elif sys.platform.startswith('darwin'):
        # macOS系统优先使用苹方和华文黑体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti'] + plt.rcParams['font.sans-serif']
    elif sys.platform.startswith('linux'):
        # Linux系统优先使用文泉驿和Noto字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']

    print("字体配置完成。使用系统适配的中文字体。")

    return True

def format_chinese_text(text):
    """
    格式化中文文本以确保正确显示。
    使用matplotlib的内置文本渲染能力，同时提供备选方案。

    参数:
        text (str): 要格式化的文本，可能包含中文字符

    返回:
        str: 格式化后的文本
    """
    # 检查是否有中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

    if not has_chinese:
        # 如果没有中文字符，直接返回原文本
        return text

    # 对于包含中文的文本，确保使用正确的字体
    # 这里我们不做特殊处理，因为已经在setup_font中配置了全局字体
    # 但如果遇到特别难以显示的文本，可以考虑使用以下备选方案之一

    return text

def create_text_as_image(text, fontsize=12, color='black'):
    """
    将文本转换为图像，然后可以作为图像嵌入到图表中。
    这是处理中文显示问题的备选方案。

    参数:
        text (str): 要转换的文本
        fontsize (int): 字体大小
        color (str): 文本颜色

    返回:
        numpy.ndarray: 包含文本的图像数组
    """
    # 创建一个临时图形
    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_subplot(111)

    # 关闭坐标轴
    ax.axis('off')

    # 添加文本
    ax.text(0.5, 0.5, text, fontsize=fontsize, color=color,
            ha='center', va='center', transform=ax.transAxes)

    # 渲染图形到内存
    fig.canvas.draw()

    # 获取图像数据
    image_data = np.array(fig.canvas.renderer.buffer_rgba())

    # 关闭图形
    plt.close(fig)

    return image_data

def add_text_to_plot(ax, x, y, text, **kwargs):
    """
    向图表添加文本，优雅地处理中文字符。

    参数:
        ax: 要添加文本的matplotlib坐标轴
        x, y: 文本的位置
        text: 要添加的文本，可能包含中文字符
        **kwargs: 传递给ax.text()的其他参数
    """
    # 格式化文本以确保中文字符正确显示
    formatted_text = format_chinese_text(text)

    # 检查是否有中文字符
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in formatted_text)

    if has_chinese:
        # 对于中文文本，确保使用正确的字体
        if 'fontproperties' not in kwargs:
            # 尝试获取一个支持中文的字体
            try:
                # 首先尝试系统字体
                if sys.platform.startswith('win'):
                    font = FontProperties(family='Microsoft YaHei')
                elif sys.platform.startswith('darwin'):
                    font = FontProperties(family='PingFang SC')
                else:
                    font = FontProperties(family='WenQuanYi Micro Hei')
                kwargs['fontproperties'] = font
            except:
                # 如果找不到合适的字体，使用默认配置
                pass

    # 将文本添加到图表
    ax.text(x, y, formatted_text, **kwargs)

    return ax

def add_text_image_to_plot(ax, x, y, text, fontsize=12, color='black'):
    """
    将文本转换为图像，然后添加到图表中。
    这是处理中文显示问题的备选方案。

    参数:
        ax: 要添加文本的matplotlib坐标轴
        x, y: 文本图像的位置（图表坐标）
        text: 要添加的文本
        fontsize: 字体大小
        color: 文本颜色
    """
    # 创建文本图像
    text_image = create_text_as_image(text, fontsize, color)

    # 将图像添加到图表
    # 注意：这需要调整图像的位置和大小
    # 这里使用的是一个简化的实现
    ax.imshow(text_image, extent=[x, x+1, y, y+1], aspect='auto', zorder=10)

    return ax

