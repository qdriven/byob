import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.ndimage import shift
from target_visualization import identify_targets, setup_font


def shift_target(data, dx, dy):
    """
    将目标按照经度方向dx和纬度方向dy进行平移

    参数:
    -----------
    data : 2D数组
        目标数据
    dx : 浮点数
        经度方向的平移距离（正值向东，负值向西）
    dy : 浮点数
        纬度方向的平移距离（正值向北，负值向南）

    返回:
    --------
    shifted_data : 2D数组
        平移后的目标数据
    """
    # 注意：在NumPy数组中，第一个维度是行（纬度），第二个维度是列（经度）
    # 因此，平移顺序应为(dy, dx)
    shifted_data = shift(data, (dy, dx), mode='constant', cval=0)

    # 打印调试信息，确保平移方向正确
    print(f"DEBUG: 平移目标 - dx={dx}° (正值向东，负值向西), dy={dy}° (正值向北，负值向南)")
    print(f"DEBUG: 在数组中实际平移为 ({dy}, {dx})")

    return shifted_data


def calculate_variance(target1, target2):
    """
    计算两个目标的匹配度，考虑重叠区域的方差和重叠程度

    参数:
    -----------
    target1 : 2D数组
        第一个目标数据
    target2 : 2D数组
        第二个目标数据

    返回:
    --------
    match_score : 浮点数
        匹配分数，越小表示匹配越好
    """
    # 计算两个目标的有效区域（大于0的区域）
    mask1 = target1 > 0
    mask2 = target2 > 0

    # 计算重叠区域
    overlap = np.logical_and(mask1, mask2)

    # 如果没有重叠，返回无穷大
    if not np.any(overlap):
        return float('inf')

    # 计算重叠区域的方差
    values1 = target1[overlap]
    values2 = target2[overlap]
    value_diff_var = np.mean((values1 - values2) ** 2)  # 使用均方误差而非方差

    # 计算重叠程度
    union = np.logical_or(mask1, mask2)
    overlap_ratio = np.sum(overlap) / np.sum(union)

    # 综合考虑值的差异和重叠程度
    # 重叠程度越高，分数越低（越好）
    # 值的差异越小，分数越低（越好）
    match_score = value_diff_var * (1.0 - overlap_ratio)

    return match_score


def find_optimal_shift(model_target, obs_target, max_shift_x=10, max_shift_y=15, step=0.1):
    """
    寻找使模式目标与观测目标之间方差最小的最优平移距离
    使用自适应搜索策略：先粗略搜索，再精细搜索

    参数:
    -----------
    model_target : 2D数组
        模式目标降水数据
    obs_target : 2D数组
        观测目标降水数据
    max_shift_x : 浮点数
        经度方向最大平移距离（单位：度）
    max_shift_y : 浮点数
        纬度方向最大平移距离（单位：度）
    step : 浮点数
        精细搜索的步长（单位：度）

    返回:
    --------
    best_shift : 元组 (dx, dy)
        经度和纬度方向的最优平移距离
    best_variance : 浮点数
        达到的最小方差
    """
    # 第一阶段：粗略搜索
    coarse_step = 1.0  # 粗略搜索步长
    best_variance = float('inf')
    best_shift = (0, 0)

    print(f"\n第一阶段: 粗略搜索 (步长={coarse_step}°)")

    for dx in np.arange(-max_shift_x, max_shift_x + coarse_step, coarse_step):
        for dy in np.arange(-max_shift_y, max_shift_y + coarse_step, coarse_step):
            shifted = shift_target(model_target, dx, dy)
            variance = calculate_variance(shifted, obs_target)

            if variance < best_variance:
                best_variance = variance
                best_shift = (dx, dy)

    # 将平移表示为四个方向
    east_west = best_shift[0]
    north_south = best_shift[1]

    east_west_dir = "东" if east_west >= 0 else "西"
    north_south_dir = "北" if north_south >= 0 else "南"

    east_west_val = abs(east_west)
    north_south_val = abs(north_south)

    print(
        f"粗略搜索最优平移: 向{east_west_dir} {east_west_val:.1f}°, 向{north_south_dir} {north_south_val:.1f}°, 方差: {best_variance:.4f}")

    # 第二阶段：在最优解附近进行精细搜索
    dx_best, dy_best = best_shift
    search_range = coarse_step * 1.5  # 在粗略步长的1.5倍范围内进行精细搜索

    print(f"\n第二阶段: 精细搜索 (步长={step}°)")
    print(
        f"搜索范围: 经度 [{dx_best - search_range:.1f}, {dx_best + search_range:.1f}], 纬度 [{dy_best - search_range:.1f}, {dy_best + search_range:.1f}]")

    for dx in np.arange(dx_best - search_range, dx_best + search_range + step, step):
        for dy in np.arange(dy_best - search_range, dy_best + search_range + step, step):
            # 确保在最大范围内
            if abs(dx) <= max_shift_x and abs(dy) <= max_shift_y:
                shifted = shift_target(model_target, dx, dy)
                variance = calculate_variance(shifted, obs_target)

                if variance < best_variance:
                    best_variance = variance
                    best_shift = (dx, dy)

    # 将平移表示为四个方向
    east_west = best_shift[0]
    north_south = best_shift[1]

    east_west_dir = "东" if east_west >= 0 else "西"
    north_south_dir = "北" if north_south >= 0 else "南"

    east_west_val = abs(east_west)
    north_south_val = abs(north_south)

    print(
        f"精细搜索最优平移: 向{east_west_dir} {east_west_val:.2f}°, 向{north_south_dir} {north_south_val:.2f}°, 方差: {best_variance:.4f}")

    return best_shift, best_variance


def plot_comparison(obs_target, model_target_shifted, lon, lat, save_path=None,
                    title="观测与平移后模式降水对比", date=None, threshold=25):
    """
    绘制观测目标（原始颜色和格点）和平移后的模式目标（等值线）对比图

    参数:
    -----------
    obs_target : 2D数组
        观测目标数据
    model_target_shifted : 2D数组
        平移后的模式目标数据
    lon : 1D数组
        经度数组
    lat : 1D数组
        纬度数组
    save_path : 字符串，可选
        保存路径
    title : 字符串，可选
        图像标题
    date : datetime对象，可选
        日期
    threshold : 浮点数，可选
        降水阈值
    """
    # 设置中文字体
    setup_font()

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # 创建绘图网格
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # 设置图形背景为白色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # 设置降水等级和颜色
    levels = [0.1, 10, 25, 50, 100, 250, 500]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 绘制观测目标的填充等值图
    cf = ax.contourf(lon_mesh, lat_mesh, obs_target, levels=levels,
                     cmap=cmap, norm=norm, extend='both', alpha=0.7, zorder=5)

    # 获取目标区域的掩码
    target_mask = obs_target >= threshold

    # 获取目标区域的坐标
    y_indices, x_indices = np.where(target_mask)

    # 转换为经纬度坐标
    target_lons = lon[x_indices]
    target_lats = lat[y_indices]

    # 绘制标记点
    ax.scatter(target_lons, target_lats, color='red', marker='s', s=15,
               alpha=0.8, edgecolor='none', zorder=10)

    # 绘制模式数据：显示阈值的等值线，使用蓝色
    cs = ax.contour(lon_mesh, lat_mesh, model_target_shifted,
                    levels=[threshold], colors='blue', linewidths=2.0, zorder=10)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%g', colors='blue')

    # 计算重叠区域
    overlap_mask = np.logical_and(obs_target >= threshold, model_target_shifted >= threshold)

    # 如果有重叠区域，用黄色高亮显示
    if np.any(overlap_mask):
        y_indices, x_indices = np.where(overlap_mask)
        overlap_lons = lon[x_indices]
        overlap_lats = lat[y_indices]
        ax.scatter(overlap_lons, overlap_lats, color='yellow', marker='s', s=20,
                   alpha=1.0, edgecolor='black', linewidth=0.5, zorder=15)

    # 添加地图要素
    ax.coastlines(linewidth=0.8, zorder=15)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False

    # 使用中文标题
    if date:
        date_str = date.strftime("%Y-%m-%d")
        title_text = f"{title} (阈值={threshold:.1f}mm) - {date_str}"
    else:
        title_text = f"{title} (阈值={threshold:.1f}mm)"

    ax.set_title(title_text)

    # 添加颜色条
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('降水量 (mm)')

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8,
               label=f'观测目标格点 (阈值≥{threshold:.1f}mm)'),
        Line2D([0], [0], color='blue', lw=1.5, label=f'平移后模式目标 = {threshold:.1f}mm')
    ]

    # 如果有重叠区域，添加到图例
    if np.any(overlap_mask):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow',
                   markeredgecolor='black', markersize=8,
                   label=f'重叠区域 ({np.sum(overlap_mask)} 个格点)')
        )

    ax.legend(handles=legend_elements, loc='lower right')

    # 保存图像前，检查是否有观测数据超过阈值
    if np.sum(target_mask) == 0:
        print(f"警告: 观测数据中没有超过{threshold}mm的点，图中将不会显示观测目标格点")
        # 在图上添加警告文本
        ax.text(0.5, 0.5, f"观测数据中没有超过{threshold}mm的降水",
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False)
        plt.close()
    else:
        plt.show()


def perform_target_shift_analysis(obs_data, model_data, date,
                                  obs_target_idx=3, model_target_idx=0,
                                  threshold=25, min_size=8,
                                  max_shift_x=10, max_shift_y=15, step=0.1,
                                  output_folder='output/target_shift',
                                  auto_adjust=True, force_use_available=True):
    """
    执行目标平移分析

    参数:
    -----------
    obs_data : 字典
        观测数据字典
    model_data : 字典
        模式数据字典
    date : datetime对象
        分析日期
    obs_target_idx : 整数
        观测目标索引（从0开始）
    model_target_idx : 整数
        模式目标索引（从0开始）
    threshold : 浮点数
        降水阈值
    min_size : 整数
        目标最小尺寸
    max_shift_x : 浮点数
        经度方向最大平移距离
    max_shift_y : 浮点数
        纬度方向最大平移距离
    step : 浮点数
        平移搜索步长
    output_folder : 字符串
        输出文件夹路径
    auto_adjust : 布尔值
        是否自动调整参数以找到足够的目标
    # 自适应搜索策略现在是默认行为，不再需要参数
    force_use_available : 布尔值
        当请求的目标索引超出范围时，是否使用可用的目标

    返回:
    --------
    best_shift : 元组 (dx, dy)
        最优平移距离
    best_variance : 浮点数
        最小方差
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 保存原始参数，用于可能的自动调整
    original_threshold = threshold
    original_min_size = min_size
    current_threshold = threshold
    current_min_size = min_size

    # 识别观测和模式数据中的目标
    print(f"识别观测数据中的降水目标（阈值={current_threshold}mm，最小尺寸={current_min_size}）...")
    obs_targets = identify_targets(obs_data, threshold=current_threshold, min_size=current_min_size, separate=True)

    print(f"识别模式数据中的降水目标（阈值={current_threshold}mm，最小尺寸={current_min_size}）...")
    model_targets = identify_targets(model_data, threshold=current_threshold, min_size=current_min_size, separate=True)

    # 打印目标数量
    print(f"找到 {len(obs_targets)} 个观测目标和 {len(model_targets)} 个模式目标")

    # 如果启用自动调整且目标不足，尝试调整参数
    if auto_adjust and (len(obs_targets) <= obs_target_idx or len(model_targets) <= model_target_idx):
        print("\n目标数量不足，尝试自动调整参数...")

        # 尝试减小最小尺寸
        while current_min_size > 3 and (len(obs_targets) <= obs_target_idx or len(model_targets) <= model_target_idx):
            current_min_size -= 1
            print(f"减小最小尺寸至 {current_min_size}，重新识别目标...")

            obs_targets = identify_targets(obs_data, threshold=current_threshold, min_size=current_min_size,
                                           separate=True)
            model_targets = identify_targets(model_data, threshold=current_threshold, min_size=current_min_size,
                                             separate=True)

            print(f"找到 {len(obs_targets)} 个观测目标和 {len(model_targets)} 个模式目标")

        # 如果减小最小尺寸后仍然不足，尝试降低阈值
        if len(obs_targets) <= obs_target_idx or len(model_targets) <= model_target_idx:
            # 重置最小尺寸
            current_min_size = original_min_size

            # 逐步降低阈值
            while current_threshold > 10 and (
                    len(obs_targets) <= obs_target_idx or len(model_targets) <= model_target_idx):
                current_threshold -= 5
                print(f"降低阈值至 {current_threshold}mm，重新识别目标...")

                obs_targets = identify_targets(obs_data, threshold=current_threshold, min_size=current_min_size,
                                               separate=True)
                model_targets = identify_targets(model_data, threshold=current_threshold, min_size=current_min_size,
                                                 separate=True)

                print(f"找到 {len(obs_targets)} 个观测目标和 {len(model_targets)} 个模式目标")

    # 打印目标编号与索引对应关系
    print("\n观测目标编号与索引对应关系:")
    for i, _ in enumerate(obs_targets):
        print(f"  观测目标{i + 1} -> 索引{i}")

    print("\n模式目标编号与索引对应关系:")
    for i, _ in enumerate(model_targets):
        print(f"  模式目标{i + 1}s -> 索引{i}")

    # 检查观测目标索引是否有效
    if len(obs_targets) <= obs_target_idx:
        if force_use_available and len(obs_targets) > 0:
            print(f"警告：观测目标索引 {obs_target_idx} 超出范围（0-{len(obs_targets) - 1}）")
            print(f"将使用最后一个可用的观测目标（索引{len(obs_targets) - 1}）")
            obs_target_idx = len(obs_targets) - 1
        else:
            print(f"错误：观测目标索引 {obs_target_idx} 超出范围（0-{len(obs_targets) - 1}）")
            return None, None

    # 检查模式目标索引是否有效
    if len(model_targets) <= model_target_idx:
        if force_use_available and len(model_targets) > 0:
            print(f"警告：模式目标索引 {model_target_idx} 超出范围（0-{len(model_targets) - 1}）")
            print(f"将使用最后一个可用的模式目标（索引{len(model_targets) - 1}）")
            model_target_idx = len(model_targets) - 1
        else:
            print(f"错误：模式目标索引 {model_target_idx} 超出范围（0-{len(model_targets) - 1}）")
            return None, None

    # 获取指定的观测和模式目标
    obs_target = obs_targets[obs_target_idx]
    model_target = model_targets[model_target_idx]

    # 计算目标的显示索引（从1开始）
    obs_display_idx = obs_target_idx + 1
    model_display_idx = model_target_idx + 1

    # 寻找最优平移
    print(f"寻找模式目标{model_display_idx}s向观测目标{obs_display_idx}的最优平移...")
    print(f"经度方向最大平移={max_shift_x}°，纬度方向最大平移={max_shift_y}°，步长={step}°")
    print(f"自适应搜索: 是")

    best_shift, best_variance = find_optimal_shift(
        model_target, obs_target, max_shift_x=max_shift_x, max_shift_y=max_shift_y, step=step
    )

    # 应用最优平移
    model_target_shifted = shift_target(model_target, *best_shift)

    # 绘制对比图
    date_str = date.strftime("%Y%m%d")

    # 如果参数被自动调整，在文件名中反映这一点
    param_suffix = ""
    if current_threshold != original_threshold:
        param_suffix += f"_阈值{current_threshold}"
    if current_min_size != original_min_size:
        param_suffix += f"_最小尺寸{current_min_size}"

    save_path = os.path.join(output_folder,
                             f'目标平移_观测{obs_display_idx}_模式{model_display_idx}s_{date_str}{param_suffix}.png')

    # 如果参数被自动调整，在标题中反映这一点
    title_suffix = ""
    if current_threshold != original_threshold or current_min_size != original_min_size:
        title_suffix = f" (自动调整: 阈值={current_threshold}mm, 最小尺寸={current_min_size})"

    plot_comparison(
        obs_target, model_target_shifted,
        obs_data['lon'], obs_data['lat'],
        save_path=save_path,
        title=f"观测目标{obs_display_idx}与平移后的模式目标{model_display_idx}s对比{title_suffix}",
        date=date,
        threshold=current_threshold
    )

    # 打印结果
    print(f"\n{date.strftime('%Y-%m-%d')} 的分析结果:")
    print(f"观测目标: {obs_display_idx}, 模式目标: {model_display_idx}s")
    if current_threshold != original_threshold or current_min_size != original_min_size:
        print(f"参数自动调整: 阈值={current_threshold}mm, 最小尺寸={current_min_size}")
    # 将平移表示为四个方向
    east_west = best_shift[0]
    north_south = best_shift[1]

    east_west_dir = "东" if east_west >= 0 else "西"
    north_south_dir = "北" if north_south >= 0 else "南"

    east_west_val = abs(east_west)
    north_south_val = abs(north_south)

    print(f"最优平移: 向{east_west_dir} {east_west_val:.2f}°, 向{north_south_dir} {north_south_val:.2f}°")
    print(f"最小方差: {best_variance:.2f}")
    print(f"可视化结果保存在: {save_path}")

    return best_shift, best_variance


# 示例用法
if __name__ == "__main__":
    from data_process import read_observation_data, read_model_data, interpolate_to_grid

    # 设置数据路径
    obs_data_path = '观测数据202107.txt'
    model_data_folder = '模式数据/'
    output_folder = 'output/target_shift'

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 目标日期
    target_date = datetime(2021, 7, 18)

    # 读取数据
    print("读取观测数据...")
    daily_rainfall = read_observation_data(obs_data_path)

    print("插值到规则网格...")
    grid_data = interpolate_to_grid(daily_rainfall)

    print("读取模式数据...")
    model_data = read_model_data(model_data_folder)

    # 获取目标日期的数据
    if target_date not in grid_data:
        print(f"错误：{target_date} 没有观测数据")
        exit(1)

    if target_date not in model_data:
        print(f"错误：{target_date} 没有模式数据")
        exit(1)

    obs_data = grid_data[target_date]
    model_data_day = model_data[target_date]

    # 执行目标平移分析
    # 默认使用观测目标4（索引3）和模式目标1s（索引0）
    best_shift, best_variance = perform_target_shift_analysis(
        obs_data, model_data_day, target_date,
        obs_target_idx=3,  # 观测目标4（从0开始索引）
        model_target_idx=0,  # 模式目标1s（从0开始索引）
        threshold=25,
        min_size=8,
        max_shift=5,
        step=0.1,
        output_folder=output_folder
    )
