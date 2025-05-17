
import os
from data_process import (
    read_observation_data,
    interpolate_to_grid,
    read_model_data
)
from visualization import (
    plot_rainfall_contour,
    plot_observation_vs_model_improved,
    identify_and_plot_obs_clusters,
    identify_and_plot_model_clusters
)
from target_visualization import (
    identify_targets,
    plot_targets_with_markers,
    plot_observation_vs_model_with_markers
)


# 在main.py中，不需要加载geopandas，直接传递shapefile路径
def main():
    # 设置数据路径
    obs_data_path = '观测数据202107.txt'  # 请替换为实际路径
    model_data_folder = '模式数据/'  # 请替换为实际路径
    output_folder = 'output/'
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 1. 读取观测数据
    print("读取观测数据...")
    daily_rainfall = read_observation_data(obs_data_path)

    # 2. 将观测数据插值到规则网格
    print("插值到规则网格...")
    grid_data = interpolate_to_grid(daily_rainfall)

    # 3. 读取模式数据
    print("读取模式数据...")
    model_data = read_model_data(model_data_folder)

    # 4. 可视化
    print("生成可视化结果...")
    # 遍历所有日期
    for date in grid_data.keys():
        date_str = date.strftime("%Y%m%d")

        # 4.1 绘制单日观测数据等值图
        obs_save_path = os.path.join(output_folder, f'单日观测降水_{date_str}.png')
        plot_rainfall_contour(grid_data[date], date, save_path=obs_save_path, 
                             title=f'单日观测降水 - {date.strftime("%Y-%m-%d")}')

        # 4.2 如果有对应的模式数据，绘制观测与模拟对比图
        if date in model_data:
            compare_save_path = os.path.join(output_folder, f'观测模拟对比_{date_str}.png')
            plot_observation_vs_model_improved(
                {date: grid_data[date]},  # 观测数据
                {date: model_data[date]},  # 模拟数据
                date,
                save_path=compare_save_path,
                title=f'观测与模拟降水对比 - {date.strftime("%Y-%m-%d")}'
            )
            
            # 4.3 使用新的可视化函数，绘制带有目标标记的图
            # 创建目标可视化输出文件夹
            target_vis_folder = os.path.join(output_folder, '目标可视化')
            os.makedirs(target_vis_folder, exist_ok=True)
            
            # 识别观测和模式数据中的目标
            obs_targets = identify_targets(grid_data[date], threshold=25, min_size=8, separate=True)
            model_targets = identify_targets(model_data[date], threshold=25, min_size=8, separate=True)
            
            # 绘制带有目标标记的观测数据图
            obs_markers_path = os.path.join(target_vis_folder, f'观测目标标记图_{date_str}.png')
            plot_targets_with_markers(
                grid_data[date], obs_targets, grid_data[date]['lon'], grid_data[date]['lat'],
                save_path=obs_markers_path,
                title="观测降水目标标记图", date=date
            )
            
            # 绘制带有目标标记的模式数据图
            model_markers_path = os.path.join(target_vis_folder, f'模式目标标记图_{date_str}.png')
            plot_targets_with_markers(
                model_data[date], model_targets, model_data[date]['lon'], model_data[date]['lat'],
                save_path=model_markers_path,
                title="模式降水目标标记图", date=date
            )
            
            # 绘制观测与模式对比图（带目标标记）
            compare_markers_path = os.path.join(target_vis_folder, f'观测模式目标对比图_{date_str}.png')
            plot_observation_vs_model_with_markers(
                grid_data[date], obs_targets, model_data[date], model_targets, date,
                save_path=compare_markers_path
            )
            
            print(f"已生成带目标标记的可视化结果，保存在 {target_vis_folder} 文件夹中")

    # 5. 单独进行目标识别
    print("\n进行降水目标识别...")
    
    # 创建目标识别输出文件夹
    target_folder = os.path.join(output_folder, '降水目标')
    os.makedirs(target_folder, exist_ok=True)
    
    # 5.1 观测数据目标识别
    print("\n识别观测数据中的降水目标...")
    obs_clusters_path = os.path.join(target_folder, '观测数据中的降水目标.png')
    obs_clusters = identify_and_plot_obs_clusters(
        grid_data, threshold=25, min_size=8,
        save_path=obs_clusters_path
    )
    
    # 5.2 模式数据目标识别
    print("\n识别模式数据中的降水目标...")
    model_clusters_path = os.path.join(target_folder, '模式数据中的降水目标.png')
    model_clusters = identify_and_plot_model_clusters(
        model_data, threshold=25, min_size=8,
        save_path=model_clusters_path
    )

    print(f"\n分析完成！结果保存在 {output_folder} 文件夹中")
    print(f"降水目标识别结果保存在 {target_folder} 文件夹中")
    print(f"带目标标记的可视化结果保存在 {os.path.join(output_folder, '目标可视化')} 文件夹中")


if __name__ == "__main__":
    main()
