import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
from datetime import datetime


def read_full_observation_data(file_path, targe_year=2021, target_month=7):
    """
    读取观测站点的降水数据,Read data file to dataframe
    """
    # 尝试不同的编码格式读取原始文本
    encodings = ['gbk', 'gb2312', 'latin1', 'utf-8']

    # 使用二进制模式读取文件，然后尝试不同的编码解码
    with open(file_path, 'rb') as f:
        content_bytes = f.read()

    content = None
    for encoding in encodings:
        try:
            content = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise Exception("无法读取文件，请检查文件编码")

    # 按空格分割内容，得到所有记录（包括标题）
    records = content.strip().split(' ')

    # 第一个记录应该是标题
    header = records[0].split(',')

    # 预分配内存，提高效率
    data_dict = {col: [] for col in header}

    # 解析数据记录
    for record in records[1:]:
        if record.strip():  # 跳过空记录
            fields = record.split(',')
            if len(fields) == len(header):
                for i, col in enumerate(header):
                    data_dict[col].append(fields[i])

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 一次性转换数据类型，减少循环
    numeric_cols = ['Year', 'Mon', 'Day', 'Hour', 'Lat', 'Lon', 'PRE_1h']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df
    # # 筛选2021年7月的数据
    # df = df[(df['Year'] == targe_year) & (df['Mon'] == target_month)]
    #
    # # 按站点位置和日期分组，计算日降水量
    # df['date'] = pd.to_datetime(df[['Year', 'Mon', 'Day']].astype(str).agg('-'.join, axis=1))
    # daily_rainfall = df.groupby(['Lon', 'Lat', 'date'])['PRE_1h'].sum().reset_index()
    #
    # # 重命名列以匹配后续处理
    # daily_rainfall.rename(columns={'Lon': 'lon', 'Lat': 'lat', 'PRE_1h': 'rainfall'}, inplace=True)
    #
    # return daily_rainfall


def convert_observation_to_csv(input_file, output_file):
    """
    将观测数据从文本文件转换为CSV格式，保留所有原始数据
    
    参数:
    input_file: 输入的文本文件路径
    output_file: 输出的CSV文件路径
    """
    df = read_full_observation_data(input_file)

    # 确保输出目录存在
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # 保存为CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"观测数据已成功转换为CSV格式，保存在: {output_file}")
    print(f"共转换了 {len(df)} 条记录，{len(df.columns)} 个字段")
    return True


def check_city_coverage_by_coordinates(observation_data, shapefile_path):
    """
    使用经纬度查找的方式，检查观测数据中的城市覆盖比例
    
    参数:
    observation_data: 观测数据文件路径或DataFrame
    shapefile_path: shapefile文件路径
    
    返回:
    coverage_ratio: 覆盖比例
    matched_cities: 匹配到的城市列表
    unmatched_coordinates: 未匹配到的坐标列表
    """
    try:
        # 读取观测数据
        if isinstance(observation_data, str):
            # 如果是文件路径，则读取文件
            df = read_full_observation_data(observation_data)
        else:
            # 如果已经是DataFrame，则直接使用
            df = observation_data

        # 读取shapefile
        reader = shapereader.Reader(shapefile_path)
        geometries = list(reader.geometries())
        records = list(reader.records())

        total_shapefile_cities = len(records)
        print(f"共加载了 {total_shapefile_cities} 个城市记录")

        # 提取观测数据中的经纬度信息
        coordinates_list = []
        if 'Lon' in df.columns and 'Lat' in df.columns:
            for _, row in df.iterrows():
                lon = float(row['Lon']) if not pd.isna(row['Lon']) else None
                lat = float(row['Lat']) if not pd.isna(row['Lat']) else None
                if lon is not None and lat is not None:
                    coordinates_list.append((lon, lat, row.get('City', '未知')))

            print(f"观测数据中共有 {len(coordinates_list)} 个坐标点")
        else:
            print("观测数据中没有找到'Lon'或'Lat'列")
            return 0, [], []

        # 创建点对象并检查每个坐标
        from shapely.geometry import Point

        matched_cities = []
        unmatched_coordinates = []
        city_coverage = set()  # 用于记录覆盖了哪些城市

        print(f"开始检查 {len(coordinates_list)} 个坐标点...")

        for i, (lon, lat, city_name) in enumerate(coordinates_list):
            point = Point(lon, lat)
            found = False

            # 检查点是否在任何一个几何形状内
            for record, geometry in zip(records, geometries):
                if 'name' in record.attributes and record.attributes['name'] is not None:
                    shapefile_city = record.attributes['name']

                    # 检查点是否在几何形状内
                    if geometry.contains(point):
                        matched_cities.append((lon, lat, city_name, shapefile_city))
                        city_coverage.add(shapefile_city)
                        found = True
                        break

            if not found:
                unmatched_coordinates.append((lon, lat, city_name))

            # 每处理100个坐标打印一次进度
            if (i + 1) % 100 == 0 or i == len(coordinates_list) - 1:
                print(f"已处理 {i + 1}/{len(coordinates_list)} 个坐标")

        # 计算覆盖比例
        coverage_ratio = len(city_coverage) / total_shapefile_cities * 100

        # 打印结果
        print(f"\n覆盖比例: {coverage_ratio:.2f}%")
        print(
            f"匹配到的坐标点: {len(matched_cities)}/{len(coordinates_list)} ({len(matched_cities) / len(coordinates_list) * 100:.2f}%)")
        print(f"覆盖的城市数: {len(city_coverage)}/{total_shapefile_cities} ({coverage_ratio:.2f}%)")

        # 打印未匹配的坐标点
        if unmatched_coordinates:
            print(f"\n未匹配到的坐标点: {len(unmatched_coordinates)}")
            for lon, lat, city_name in unmatched_coordinates[:10]:  # 只打印前10个
                print(f"  - ({lon}, {lat}) {city_name}")
            if len(unmatched_coordinates) > 10:
                print(f"  ... 还有 {len(unmatched_coordinates) - 10} 个未显示")

        return coverage_ratio, matched_cities, unmatched_coordinates

    except Exception as e:
        print(f"检查城市覆盖比例时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], []


def check_city_coverage_from_ok_geo(observation_data, ok_geo_csv_path):
    """
    从ok_geo.csv文件检查观测数据中的城市覆盖比例
    
    参数:
    observation_data: 观测数据文件路径或DataFrame
    ok_geo_csv_path: ok_geo.csv文件路径
    
    返回:
    coverage_ratio: 覆盖比例
    matched_cities: 匹配到的城市列表
    unmatched_coordinates: 未匹配到的坐标列表
    """
    try:
        # 读取观测数据
        if isinstance(observation_data, str):
            # 如果是文件路径，则读取文件
            df = read_full_observation_data(observation_data)
        else:
            # 如果已经是DataFrame，则直接使用
            df = observation_data

        # 尝试使用不同的参数读取ok_geo.csv文件
        try:
            # 方法1：使用引擎C，但设置on_bad_lines参数为'skip'
            geo_df = pd.read_csv(ok_geo_csv_path, encoding='utf-8', on_bad_lines='skip')
        except:
            try:
                # 方法2：使用Python引擎，通常更宽容但速度较慢
                geo_df = pd.read_csv(ok_geo_csv_path, encoding='utf-8', engine='python')
            except:
                try:
                    # 方法3：尝试不同的编码
                    geo_df = pd.read_csv(ok_geo_csv_path, encoding='latin1', engine='python')
                except:
                    # 方法4：手动读取文件，逐行解析
                    print("尝试手动读取文件...")
                    with open(ok_geo_csv_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()

                    # 获取标题行
                    header = lines[0].strip().split(',')

                    # 准备数据
                    data = []
                    for i, line in enumerate(lines[1:], 1):
                        try:
                            # 尝试解析每一行
                            row = line.strip().split(',')
                            # 确保行数据与标题列数匹配
                            if len(row) >= len(header):
                                data.append(row[:len(header)])
                            else:
                                # 如果列数不匹配，填充空值
                                data.append(row + [''] * (len(header) - len(row)))
                        except Exception as e:
                            print(f"跳过第 {i + 1} 行: {e}")

                    # 创建DataFrame
                    geo_df = pd.DataFrame(data, columns=header)

        print(f"从 {ok_geo_csv_path} 加载了 {len(geo_df)} 条地理数据记录")

        # 检查必要的列是否存在
        required_cols = ['name', 'geo', 'polygon']  # 修改为实际的列名
        missing_cols = [col for col in required_cols if col not in geo_df.columns]
        if missing_cols:
            print(f"警告: ok_geo.csv 缺少必要的列: {', '.join(missing_cols)}")
            print(f"可用的列: {', '.join(geo_df.columns)}")
            return 0, [], []

        # 提取观测数据中的经纬度信息
        coordinates_list = []
        if 'Lon' in df.columns and 'Lat' in df.columns:
            for _, row in df.iterrows():
                lon = float(row['Lon']) if not pd.isna(row['Lon']) else None
                lat = float(row['Lat']) if not pd.isna(row['Lat']) else None
                if lon is not None and lat is not None:
                    coordinates_list.append((lon, lat, row.get('City', '未知')))

            print(f"观测数据中共有 {len(coordinates_list)} 个坐标点")
        else:
            print("观测数据中没有找到'Lon'或'Lat'列")
            return 0, [], []

        # 从ok_geo.csv中提取城市边界多边形
        from shapely.geometry import Point, Polygon
        import json

        city_polygons = []
        for _, row in geo_df.iterrows():
            try:
                city_name = row['name']
                polygon_str = row['polygon'] if 'polygon' in row else None
                geo_str = row['geo'] if 'geo' in row else None

                # 优先使用polygon列，如果没有则尝试使用geo列
                coord_str = polygon_str if polygon_str and str(polygon_str).strip() else geo_str

                if not coord_str or pd.isna(coord_str) or str(coord_str).strip() == '':
                    continue

                # 解析多边形坐标
                try:
                    # 尝试解析JSON格式的多边形
                    if isinstance(coord_str, str):
                        # 如果是WKT格式 (例如: "POLYGON((x1 y1, x2 y2, ...))")
                        if coord_str.upper().startswith('POLYGON'):
                            from shapely import wkt
                            polygon = wkt.loads(coord_str)
                            city_polygons.append((city_name, polygon))
                        else:
                            # 尝试解析为JSON
                            try:
                                polygon_data = json.loads(coord_str)
                                # 根据多边形数据格式进行处理
                                if isinstance(polygon_data, list):
                                    # 如果是坐标列表
                                    if len(polygon_data) > 0 and isinstance(polygon_data[0], list):
                                        polygon = Polygon(polygon_data)
                                        city_polygons.append((city_name, polygon))
                            except json.JSONDecodeError:
                                # 如果不是JSON格式，尝试其他格式解析
                                # 例如: "x1,y1 x2,y2 x3,y3"
                                try:
                                    coords = []
                                    for coord_pair in coord_str.split():
                                        x, y = map(float, coord_pair.split(','))
                                        coords.append((x, y))
                                    if len(coords) >= 3:  # 至少需要3个点才能形成多边形
                                        polygon = Polygon(coords)
                                        city_polygons.append((city_name, polygon))
                                except Exception as e:
                                    pass
                except Exception as e:
                    print(f"无法解析城市 {city_name} 的多边形数据: {e}")
            except Exception as e:
                print(f"处理城市记录时出错: {e}")

        print(f"成功解析了 {len(city_polygons)} 个城市的多边形边界")

        # 检查每个坐标点是否在任何城市多边形内
        matched_cities = []
        unmatched_coordinates = []
        city_coverage = set()  # 用于记录覆盖了哪些城市

        print(f"开始检查 {len(coordinates_list)} 个坐标点...")

        for i, (lon, lat, city_name) in enumerate(coordinates_list):
            point = Point(lon, lat)
            found = False

            # 检查点是否在任何一个城市多边形内
            for geo_city_name, polygon in city_polygons:
                try:
                    if polygon.contains(point):
                        matched_cities.append((lon, lat, city_name, geo_city_name))
                        city_coverage.add(geo_city_name)
                        found = True
                        break
                except Exception as e:
                    print(f"检查点 ({lon}, {lat}) 是否在城市 {geo_city_name} 内时出错: {e}")

            if not found:
                unmatched_coordinates.append((lon, lat, city_name))

            # 每处理100个坐标打印一次进度
            if (i + 1) % 100 == 0 or i == len(coordinates_list) - 1:
                print(f"已处理 {i + 1}/{len(coordinates_list)} 个坐标")

        # 计算覆盖比例
        total_cities = len(city_polygons)
        coverage_ratio = len(city_coverage) / total_cities * 100 if total_cities > 0 else 0

        # 打印结果
        print(f"\n覆盖比例: {coverage_ratio:.2f}%")
        print(
            f"匹配到的坐标点: {len(matched_cities)}/{len(coordinates_list)} ({len(matched_cities) / len(coordinates_list) * 100:.2f}%)")
        print(f"覆盖的城市数: {len(city_coverage)}/{total_cities} ({coverage_ratio:.2f}%)")

        # 打印未匹配的坐标点
        if unmatched_coordinates:
            print(f"\n未匹配到的坐标点: {len(unmatched_coordinates)}")
            for lon, lat, city_name in unmatched_coordinates[:10]:  # 只打印前10个
                print(f"  - ({lon}, {lat}) {city_name}")
            if len(unmatched_coordinates) > 10:
                print(f"  ... 还有 {len(unmatched_coordinates) - 10} 个未显示")

        return coverage_ratio, matched_cities, unmatched_coordinates

    except Exception as e:
        print(f"从ok_geo.csv检查城市覆盖比例时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], []


if __name__ == "__main__":
    # 设置数据路径
    obs_data_path = '观测数据202107.txt'
    city_shapefile = 'ok_data_level3/ok_data_level3.shp'
    ok_geo_csv_path = 'data/ok_geo.csv'  # 修正路径
    output_folder = 'output'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 转换观测数据为CSV
    csv_output_path = os.path.join(output_folder, '观测数据202107.csv')
    convert_observation_to_csv(obs_data_path, csv_output_path)

    # 使用ok_geo.csv检查城市覆盖比例
    print("\n开始使用ok_geo.csv检查城市覆盖比例...")
    geo_coverage_ratio, geo_matched_cities, geo_unmatched_coordinates = check_city_coverage_from_ok_geo(
        obs_data_path, ok_geo_csv_path
    )

    # 将结果保存到文件
    geo_matched_file = os.path.join(output_folder, 'geo_matched_coordinates.csv')
    geo_unmatched_file = os.path.join(output_folder, 'geo_unmatched_coordinates.csv')

    # 保存ok_geo.csv匹配结果
    if geo_matched_cities:
        geo_matched_df = pd.DataFrame(geo_matched_cities, columns=['经度', '纬度', '原始城市', '匹配城市'])
        geo_matched_df.to_csv(geo_matched_file, index=False, encoding='utf-8-sig')
        print(f"ok_geo.csv匹配到的坐标点已保存到: {geo_matched_file}")

    if geo_unmatched_coordinates:
        geo_unmatched_df = pd.DataFrame(geo_unmatched_coordinates, columns=['经度', '纬度', '原始城市'])
        geo_unmatched_df.to_csv(geo_unmatched_file, index=False, encoding='utf-8-sig')
        print(f"ok_geo.csv未匹配到的坐标点已保存到: {geo_unmatched_file}")

    # 生成覆盖比例报告
    report_file = os.path.join(output_folder, 'coverage_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("城市覆盖比例报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"观测数据: {obs_data_path}\n")
        f.write(f"城市数据(ok_geo.csv): {ok_geo_csv_path}\n\n")

        f.write("=== ok_geo.csv检查结果 ===\n")
        f.write(f"观测数据中的坐标点总数: {len(geo_matched_cities) + len(geo_unmatched_coordinates)}\n")
        f.write(f"匹配到的坐标点数: {len(geo_matched_cities)}\n")
        f.write(f"未匹配到的坐标点数: {len(geo_unmatched_coordinates)}\n")

        # 计算覆盖的城市数
        covered_cities = set([city[3] for city in geo_matched_cities]) if geo_matched_cities else set()
        f.write(f"覆盖的城市数: {len(covered_cities)}\n")
        f.write(f"覆盖比例: {geo_coverage_ratio:.2f}%\n\n")

        f.write("=== 结论 ===\n")
        if geo_coverage_ratio < 10:
            f.write("  覆盖比例非常低，建议使用更大范围的数据\n")
        elif geo_coverage_ratio < 50:
            f.write("  覆盖比例较低，可能需要考虑使用更大范围的数据\n")
        else:
            f.write("  覆盖比例较高，当前数据范围可能足够\n")

    print(f"覆盖比例报告已生成，保存在: {report_file}")
