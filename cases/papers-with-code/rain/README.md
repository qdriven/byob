
# 识别并绘制模式数据中的降水聚类
python main.py --identify-model-clusters

# 识别并绘制观测数据中的降水聚类
python main.py --identify-obs-clusters

# 绘制单日降水
python main.py --plot-daily


# 使用默认参数（观测目标4，模式目标1，日期2021-07-18）
python run_target_shift_analysis.py

# 指定不同的日期
python run_target_shift_analysis.py --date 2021-07-19

# 指定不同的目标
python run_target_shift_analysis.py --obs-target 1 --model-target 2

# 指定不同的阈值和最小尺寸
python run_target_shift_analysis.py --threshold 30 --min-size 10

# 组合使用多个参数
python run_target_shift_analysis.py --date 2021-07-20 --obs-target 3 --model-target 2 --threshold 20 --max-shift 15