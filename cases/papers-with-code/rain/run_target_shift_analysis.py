#!/usr/bin/env python
import argparse
from datetime import datetime
from target_shift_analysis import perform_target_shift_analysis
from data_process import read_observation_data, read_model_data, interpolate_to_grid
import os
import numpy as np

def main():
    """
    目标平移分析的命令行接口
    """
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='执行目标平移分析')

    # 数据路径参数
    parser.add_argument('--obs', type=str, default='观测数据202107.txt',
                        help='观测数据文件路径 (默认: 观测数据202107.txt)')
    parser.add_argument('--model', type=str, default='模式数据/',
                        help='模式数据文件夹路径 (默认: 模式数据/)')

    # 目标选择参数
    parser.add_argument('--obs-target', type=int, default=4,
                        help='观测目标编号，从1开始 (默认: 4)')
    parser.add_argument('--model-target', type=int, default=1,
                        help='模式目标编号，从1开始 (默认: 1)')

    # 日期参数
    parser.add_argument('--date', type=str, default='2021-07-18',
                        help='分析日期，格式为YYYY-MM-DD (默认: 2021-07-18)')

    # 分析参数
    parser.add_argument('--threshold', type=float, default=25.0,
                        help='降水阈值，单位为mm (默认: 25.0)')
    parser.add_argument('--min-size', type=int, default=8,
                        help='目标最小尺寸，单位为网格点数 (默认: 8)')

    # 平移搜索参数
    parser.add_argument('--max-shift-x', type=float, default=10.0,
                        help='经度方向最大平移距离，单位为度 (默认: 10.0)')
    parser.add_argument('--max-shift-y', type=float, default=15.0,
                        help='纬度方向最大平移距离，单位为度 (默认: 15.0)')
    parser.add_argument('--max-shift', type=float, default=None,
                        help='同时设置经度和纬度方向的最大平移距离（如果指定，将覆盖max-shift-x和max-shift-y）')
    parser.add_argument('--step', type=float, default=0.1,
                        help='平移搜索步长，单位为度 (默认: 0.1)')
    # 自适应搜索策略现在是默认行为，不再需要参数

    # 输出参数
    parser.add_argument('--output', type=str, default='output/target_shift',
                        help='输出文件夹路径 (默认: output/target_shift)')

    # 新增参数：是否使用自动调整
    parser.add_argument('--auto-adjust', action='store_true',
                        help='自动调整参数以找到足够的目标')

    # 解析命令行参数
    args = parser.parse_args()

    # 解析日期
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"错误：日期格式不正确，应为YYYY-MM-DD，例如2021-07-18")
        return

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 处理max_shift参数
    if args.max_shift is not None:
        args.max_shift_x = args.max_shift
        args.max_shift_y = args.max_shift

    # 打印分析参数
    print(f"目标平移分析参数:")
    print(f"  日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"  观测目标: {args.obs_target}")
    print(f"  模式目标: {args.model_target}")
    print(f"  降水阈值: {args.threshold} mm")
    print(f"  最小尺寸: {args.min_size} 个网格点")
    print(f"  经度方向最大平移距离: {args.max_shift_x}°")
    print(f"  纬度方向最大平移距离: {args.max_shift_y}°")
    print(f"  搜索步长: {args.step}°")
    # 自适应搜索现在是默认行为，不再需要参数
    print(f"  自适应搜索: 是")
    print(f"  自动调整: {'是' if args.auto_adjust else '否'}")
    print(f"  输出目录: {output_dir}")
    print()

    # 读取数据
    print("读取观测数据...")
    daily_rainfall = read_observation_data(args.obs)

    print("插值到规则网格...")
    grid_data = interpolate_to_grid(daily_rainfall)

    print("读取模式数据...")
    model_data = read_model_data(args.model)

    # 获取目标日期的数据
    if target_date not in grid_data:
        print(f"错误：{target_date} 没有观测数据")
        return

    if target_date not in model_data:
        print(f"错误：{target_date} 没有模式数据")
        return

    # 获取指定日期的数据
    obs_data = grid_data[target_date]
    model_data_day = model_data[target_date]

    # 直接调用perform_target_shift_analysis函数
    # 注意：命令行参数中目标编号从1开始，但函数参数中索引从0开始
    best_shift, best_variance = perform_target_shift_analysis(
        obs_data=obs_data,
        model_data=model_data_day,
        date=target_date,
        obs_target_idx=args.obs_target - 1,  # 转换为从0开始的索引
        model_target_idx=args.model_target - 1,  # 转换为从0开始的索引
        threshold=args.threshold,
        min_size=args.min_size,
        max_shift_x=args.max_shift_x,
        max_shift_y=args.max_shift_y,
        step=args.step,
        output_folder=output_dir,
        auto_adjust=args.auto_adjust
    )

    # 检查分析结果
    if best_shift is None or best_variance is None:
        print("\n分析失败。请检查目标索引是否有效，或尝试调整阈值和最小尺寸参数。")
    else:
        print("\n分析成功完成！")
        print(f"最优平移: 向东 {best_shift[0]:.2f}°, 向北 {best_shift[1]:.2f}°")
        print(f"最小方差: {best_variance:.2f}")
        print(f"可视化结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
