# 基于人工智能的船舶电气图纸布局系统实现方案

根据论文第33-54页内容，我将为您设计一个完整的实现方案，包括神经网络模型和先验规则布局程序，以实现船舶电气图纸的智能化布局布线。

## 一、系统整体架构

系统将分为两大核心模块：
1. **神经网络参数优化器**：使用PyTorch实现，负责根据图纸拓扑特征预测最优布局参数
2. **先验规则布局器**：负责根据连接关系和优化参数执行实际布局布线

## 二、实现计划

### 1. 数据预处理模块
- 连接关系转换器：将元件连接数据转换为连接关系矩阵
- 连接矩阵压缩器：使用奇异值分解(SVD)提取拓扑特征

### 2. 神经网络参数优化器
- 设计四层神经网络结构
- 实现训练和推理功能

### 3. 先验规则布局器
- 实现三种基础布局场景的规则
- 实现节点迭代法处理复杂布局

### 4. 可视化与输出模块
- 将布局结果可视化
- 提供导出CAD格式功能

## 三、详细实现方案

### 1. 数据预处理模块

```python
import numpy as np
import torch
from scipy.linalg import svd

class DataPreprocessor:
    def __init__(self, svd_components=40):
        """
        初始化数据预处理器
        Args:
            svd_components: 奇异值分解保留的特征数量
        """
        self.svd_components = svd_components
    
    def create_connection_matrix(self, connections):
        """
        创建连接关系矩阵
        Args:
            connections: 列表，每项包含(元件1, 元件2)表示连接关系
        Returns:
            连接关系矩阵
        """
        # 获取所有唯一元件ID
        all_components = set()
        for comp1, comp2 in connections:
            all_components.add(comp1)
            all_components.add(comp2)
        
        # 创建ID到索引的映射
        comp_to_idx = {comp: idx for idx, comp in enumerate(all_components)}
        n = len(all_components)
        
        # 创建连接矩阵
        conn_matrix = np.zeros((n, n))
        for comp1, comp2 in connections:
            i, j = comp_to_idx[comp1], comp_to_idx[comp2]
            conn_matrix[i, j] = 1
            conn_matrix[j, i] = 1  # 对称矩阵
        
        return conn_matrix, comp_to_idx
    
    def compress_connection_matrix(self, conn_matrix):
        """
        使用SVD压缩连接矩阵提取特征
        Args:
            conn_matrix: 连接关系矩阵
        Returns:
            压缩后的特征向量
        """
        # 执行SVD分解
        U, s, Vt = svd(conn_matrix, full_matrices=False)
        
        # 取前k个奇异值对应的特征
        k = min(self.svd_components, len(s))
        features = U[:, :k] @ np.diag(s[:k])
        
        # 将特征向量展平为一维向量
        flattened_features = features.flatten()
        
        return torch.tensor(flattened_features, dtype=torch.float32)
```

### 2. 神经网络参数优化器

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LayoutParameterOptimizer(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 64, 256, 128], output_size=2):
        """
        布局参数优化神经网络
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层神经元数量
            output_size: 输出参数数量（长宽比和间距）
        """
        super(LayoutParameterOptimizer, self).__init__()
        
        # 构建四层神经网络，与论文结构一致
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.output_layer = nn.Linear(hidden_sizes[3], output_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """前向传播"""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        x = self.output_layer(x)
        
        # 输出参数范围调整
        aspect_ratio = 0.5 + 2.0 * self.sigmoid(x[0])  # 长宽比范围：0.5-2.5
        spacing = 0.5 + 1.5 * self.sigmoid(x[1])       # 间距范围：0.5-2.0
        
        return torch.tensor([aspect_ratio, spacing])

class ParameterOptimizerTrainer:
    def __init__(self, model, learning_rate=1e-6):
        """
        参数优化器训练类
        Args:
            model: 神经网络模型
            learning_rate: 学习率
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train_batch(self, features_batch, target_params_batch):
        """
        训练一个批次
        Args:
            features_batch: 批次特征数据
            target_params_batch: 目标参数数据
        Returns:
            损失值
        """
        self.optimizer.zero_grad()
        outputs = torch.stack([self.model(x) for x in features_batch])
        loss = self.loss_fn(outputs, target_params_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
```

### 3. 先验规则布局器

```python
class Component:
    """元件类"""
    def __init__(self, id, name, width=1, height=1, position=(0, 0)):
        self.id = id
        self.name = name
        self.width = width
        self.height = height
        self.position = position
        self.connections = []  # 存储连接的元件ID

class SubGraph:
    """子图类，表示布局完成的一组元件"""
    def __init__(self, components=None, width=0, height=0, position=(0, 0)):
        self.components = components or []
        self.width = width
        self.height = height
        self.position = position
        self.connections = []  # 存储连接的元件或子图ID

class PriorRuleLayoutEngine:
    """先验规则布局引擎"""
    def __init__(self):
        self.components = {}  # 存储所有元件
        self.subgraphs = {}   # 存储所有子图
        self.connections = [] # 存储所有连接关系
    
    def load_components(self, components_data):
        """加载元件数据"""
        for comp_data in components_data:
            comp = Component(
                id=comp_data['id'],
                name=comp_data['name'],
                width=comp_data.get('width', 1),
                height=comp_data.get('height', 1)
            )
            self.components[comp.id] = comp
    
    def load_connections(self, connections_data):
        """加载连接关系数据"""
        self.connections = connections_data
        # 更新元件的连接信息
        for comp1_id, comp2_id in connections_data:
            if comp1_id in self.components:
                self.components[comp1_id].connections.append(comp2_id)
            if comp2_id in self.components:
                self.components[comp2_id].connections.append(comp1_id)
    
    def layout_component_series(self, component_ids, aspect_ratio, spacing):
        """
        元件串联场景布局
        Args:
            component_ids: 需要布局的元件ID列表
            aspect_ratio: 长宽比
            spacing: 间距
        Returns:
            布局完成的子图
        """
        n = len(component_ids)
        
        # 计算行列数
        if aspect_ratio >= 1:
            cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
            rows = int(np.ceil(n / cols))
        else:
            rows = int(np.ceil(np.sqrt(n / aspect_ratio)))
            cols = int(np.ceil(n / rows))
        
        # 优化行列数
        if (rows - 1) * cols >= n:
            rows -= 1
        if (cols - 1) * rows >= n:
            cols -= 1
        
        # 布局元件（S型排列）
        positions = {}
        for i, comp_id in enumerate(component_ids):
            row = i // cols
            col = i % cols if row % 2 == 0 else cols - 1 - (i % cols)  # S型排列
            positions[comp_id] = (col * (1 + spacing), row * (1 + spacing))
        
        # 创建子图
        width = cols * (1 + spacing) - spacing
        height = rows * (1 + spacing) - spacing
        
        # 更新元件位置
        for comp_id, pos in positions.items():
            self.components[comp_id].position = pos
        
        # 创建并返回子图
        subgraph = SubGraph(
            components=[self.components[cid] for cid in component_ids],
            width=width,
            height=height
        )
        
        return subgraph
    
    def layout_component_subgraph_series(self, subgraph_id, component_ids, aspect_ratio, spacing):
        """
        元件子图串联场景布局
        Args:
            subgraph_id: 子图ID
            component_ids: 需要布局的元件ID列表
            aspect_ratio: 长宽比
            spacing: 间距
        Returns:
            布局完成的子图
        """
        subgraph = self.subgraphs[subgraph_id]
        
        # 子图放在最上方
        subgraph_width = subgraph.width
        
        # 计算元件布局
        n = len(component_ids)
        cols = max(1, int(subgraph_width / (1 + spacing)))
        rows = int(np.ceil(n / cols))
        
        # 布局元件（纵向S形）
        positions = {}
        for i, comp_id in enumerate(component_ids):
            row = i // cols
            col = i % cols if row % 2 == 0 else cols - 1 - (i % cols)
            positions[comp_id] = (col * (1 + spacing), subgraph.height + (1 + spacing) + row * (1 + spacing))
        
        # 更新元件位置
        for comp_id, pos in positions.items():
            self.components[comp_id].position = pos
        
        # 计算新子图尺寸
        width = max(subgraph_width, cols * (1 + spacing) - spacing)
        height = subgraph.height + (1 + spacing) + rows * (1 + spacing) - spacing
        
        # 创建并返回子图
        new_components = subgraph.components.copy()
        new_components.extend([self.components[cid] for cid in component_ids])
        
        new_subgraph = SubGraph(
            components=new_components,
            width=width,
            height=height
        )
        
        return new_subgraph
    
    def layout_component_subgraph_parallel(self, items, parent_id, aspect_ratio, spacing):
        """
        元件子图并联场景布局
        Args:
            items: 需要布局的元件和子图ID列表，格式为[(id, is_subgraph)]
            parent_id: 父节点ID
            aspect_ratio: 长宽比
            spacing: 间距
        Returns:
            布局完成的子图
        """
        # 按大小排序（先大后小）
        sorted_items = []
        for item_id, is_subgraph in items:
            if is_subgraph:
                size = self.subgraphs[item_id].width * self.subgraphs[item_id].height
                sorted_items.append((item_id, is_subgraph, size))
            else:
                sorted_items.append((item_id, is_subgraph, 1))  # 元件默认大小为1
        
        sorted_items.sort(key=lambda x: x[2], reverse=True)
        
        # 从左到右布局
        current_x = 0
        max_height = 0
        
        for item_id, is_subgraph, _ in sorted_items:
            if is_subgraph:
                subgraph = self.subgraphs[item_id]
                # 更新子图位置
                subgraph.position = (current_x, 0)
                current_x += subgraph.width + spacing
                max_height = max(max_height, subgraph.height)
            else:
                # 更新元件位置
                self.components[item_id].position = (current_x, 0)
                current_x += 1 + spacing
                max_height = max(max_height, 1)
        
        # 放置父节点在底部中央
        parent_x = (current_x - spacing) / 2 - 0.5
        parent_y = max_height + spacing + 1
        self.components[parent_id].position = (parent_x, parent_y)
        
        # 计算子图尺寸
        width = current_x - spacing
        height = parent_y + 1
        
        # 创建组件列表
        components = [self.components[parent_id]]
        for item_id, is_subgraph, _ in sorted_items:
            if is_subgraph:
                components.extend(self.subgraphs[item_id].components)
            else:
                components.append(self.components[item_id])
        
        # 创建并返回子图
        subgraph = SubGraph(
            components=components,
            width=width,
            height=height
        )
        
        return subgraph
    
    def node_iteration_layout(self, root_id, aspect_ratio, spacing):
        """
        节点迭代法布局
        Args:
            root_id: 根节点ID
            aspect_ratio: 长宽比
            spacing: 间距
        Returns:
            布局完成的子图
        """
        def is_series_components(node_id, child_ids):
            """判断是否为元件串联场景"""
            return all(child_id in self.components for child_id in child_ids)
        
        def is_series_component_subgraph(node_id, child_ids):
            """判断是否为元件子图串联场景"""
            return any(child_id in self.subgraphs for child_id in child_ids)
        
        def is_parallel_structure(node_id, child_ids):
            """判断是否为并联结构"""
            # 检查是否所有子节点都连接到同一个父节点
            for child_id in child_ids:
                if child_id in self.components:
                    connections = self.components[child_id].connections
                else:  # 子图
                    connections = self.subgraphs[child_id].connections
                
                if node_id not in connections:
                    return False
            return True
        
        def get_child_ids(node_id):
            """获取节点的子节点ID列表"""
            if node_id in self.components:
                return self.components[node_id].connections
            else:  # 子图
                return self.subgraphs[node_id].connections
        
        def layout_node(node_id):
            """递归布局节点"""
            child_ids = get_child_ids(node_id)
            
            # 如果没有子节点，返回单个元件
            if not child_ids:
                if node_id in self.components:
                    sg = SubGraph(
                        components=[self.components[node_id]],
                        width=1,
                        height=1
                    )
                    self.subgraphs[node_id] = sg
                    return node_id
                return node_id
            
            # 检查是否符合基础布局场景
            if is_series_components(node_id, child_ids):
                # 元件串联场景
                sg = self.layout_component_series(child_ids, aspect_ratio, spacing)
                sg_id = f"sg_{node_id}"
                self.subgraphs[sg_id] = sg
                return sg_id
                
            elif is_series_component_subgraph(node_id, child_ids):
                # 先处理子节点
                processed_children = []
                for child_id in child_ids:
                    processed_id = layout_node(child_id)
                    processed_children.append(processed_id)
                
                # 元件子图串联场景
                subgraph_ids = [cid for cid in processed_children if cid in self.subgraphs]
                component_ids = [cid for cid in processed_children if cid in self.components]
                
                if subgraph_ids and component_ids:
                    sg = self.layout_component_subgraph_series(
                        subgraph_ids[0], component_ids, aspect_ratio, spacing
                    )
                    sg_id = f"sg_{node_id}"
                    self.subgraphs[sg_id] = sg
                    return sg_id
                    
            elif is_parallel_structure(node_id, child_ids):
                # 先处理子节点
                processed_children = []
                for child_id in child_ids:
                    processed_id = layout_node(child_id)
                    processed_children.append((processed_id, processed_id in self.subgraphs))
                
                # 元件子图并联场景
                sg = self.layout_component_subgraph_parallel(
                    processed_children, node_id, aspect_ratio, spacing
                )
                sg_id = f"sg_{node_id}"
                self.subgraphs[sg_id] = sg
                return sg_id
            
            # 如果不符合基础场景，递归处理子节点
            processed_children = []
            for child_id in child_ids:
                processed_id = layout_node(child_id)
                processed_children.append(processed_id)
            
            # 处理完子节点后再次尝试布局
            return layout_node(node_id)
        
        # 开始递归布局
        final_sg_id = layout_node(root_id)
        return self.subgraphs[final_sg_id] if final_sg_id in self.subgraphs else None
    
    def generate_layout(self, root_id, layout_params):
        """
        生成布局
        Args:
            root_id: 根节点ID
            layout_params: 布局参数，包含aspect_ratio和spacing
        Returns:
            布局结果
        """
        aspect_ratio = layout_params[0].item()
        spacing = layout_params[1].item()
        
        # 执行节点迭代布局
        final_layout = self.node_iteration_layout(root_id, aspect_ratio, spacing)
        
        return final_layout
    
    def generate_wiring(self, layout):
        """
        生成布线
        Args:
            layout: 布局结果
        Returns:
            布线结果，包含线段列表
        """
        wires = []
        processed_connections = set()
        
        # 遍历所有连接关系
        for comp1_id, comp2_id in self.connections:
            # 避免重复处理
            if (comp1_id, comp2_id) in processed_connections or (comp2_id, comp1_id) in processed_connections:
                continue
            
            processed_connections.add((comp1_id, comp2_id))
            
            # 获取元件位置
            pos1 = None
            pos2 = None
            
            for comp in layout.components:
                if comp.id == comp1_id:
                    pos1 = comp.position
                if comp.id == comp2_id:
                    pos2 = comp.position
            
            if pos1 and pos2:
                # 创建连接线（简单直线）
                wire = {
                    'from': comp1_id,
                    'to': comp2_id,
                    'points': [pos1, pos2]
                }
                wires.append(wire)
        
        return wires
```

### 4. 可视化与输出模块

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LayoutVisualizer:
    """布局可视化器"""
    def __init__(self, layout, wiring, components):
        self.layout = layout
        self.wiring = wiring
        self.components = components
    
    def visualize(self, save_path=None):
        """
        可视化布局和布线
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制元件
        for comp in self.layout.components:
            x, y = comp.position
            rect = patches.Rectangle(
                (x, y), comp.width, comp.height, 
                linewidth=1, edgecolor='black', facecolor='lightblue'
            )
            ax.add_patch(rect)
            ax.text(x + comp.width/2, y + comp.height/2, comp.name,
                    ha='center', va='center', fontsize=8)
        
        # 绘制连线
        for wire in self.wiring:
            points = wire['points']
            x_values = [p[0] + 0.5 for p in points]  # 从元件中心连线
            y_values = [p[1] + 0.5 for p in points]
            ax.plot(x_values, y_values, 'r-', linewidth=1)
        
        # 设置图像参数
        ax.set_xlim(-1, self.layout.width + 1)
        ax.set_ylim(-1, self.layout.height + 1)
        ax.set_aspect('equal')
        ax.set_title('Ship Electrical System Layout')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def export_to_cad(self, output_path):
        """
        导出为CAD格式
        Args:
            output_path: 输出文件路径
        """
        # 这里需要实现CAD文件格式的导出
        # 可以使用第三方库如ezdxf或pyautocad
        # 简化版本，这里只导出为DXF文本格式
        with open(output_path, 'w') as f:
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            # 导出元件为矩形
            for comp in self.layout.components:
                x, y = comp.position
                f.write(f"0\nTEXT\n8\n0\n10\n{x+0.5}\n20\n{y+0.5}\n40\n0.2\n1\n{comp.name}\n")
                f.write(f"0\nRECTANGLE\n8\n0\n10\n{x}\n20\n{y}\n11\n{x+comp.width}\n21\n{y+comp.height}\n")
            
            # 导出连线
            for wire in self.wiring:
                points = wire['points']
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    f.write(f"0\nLINE\n8\n0\n10\n{x1+0.5}\n20\n{y1+0.5}\n11\n{x2+0.5}\n21\n{y2+0.5}\n")
            
            f.write("0\nENDSEC\n0\nEOF\n")
```

### 5. 主程序

```python
class ShipElectricalLayoutSystem:
    """船舶电气布局系统"""
    def __init__(self, model_path=None):
        # 初始化数据预处理器
        self.preprocessor = DataPreprocessor(svd_components=40)
        
        # 初始化神经网络模型
        input_size = 40 * 40  # SVD压缩后的特征维度
        self.model = LayoutParameterOptimizer(input_size)
        
        # 加载预训练模型
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        # 初始化布局引擎
        self.layout_engine = PriorRuleLayoutEngine()
    
    def train(self, training_data, epochs=100, batch_size=64, learning_rate=1e-6):
        """
        训练神经网络模型
        Args:
            training_data: 训练数据，包含(连接数据, 目标参数)对
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        trainer = ParameterOptimizerTrainer(self.model, learning_rate)
        
        # 准备训练数据
        features = []
        targets = []
        
        for connections, target_params in training_data:
            # 生成连接矩阵
            conn_matrix, _ = self.preprocessor.create_connection_matrix(connections)
            # 压缩特征
            feature = self.preprocessor.compress_connection_matrix(conn_matrix)
            features.append(feature)
            targets.append(torch.tensor(target_params, dtype=torch.float32))
        
        # 训练模型
        for epoch in range(epochs):
            # 打乱数据
            indices = torch.randperm(len(features))
            
            # 批次训练
            total_loss = 0
            for i in range(0, len(features), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = [features[idx] for idx in batch_indices]
                batch_targets = torch.stack([targets[idx] for idx in batch_indices])
                
                loss = trainer.train_batch(batch_features, batch_targets)
                total_loss += loss
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(features):.6f}")
        
        return trainer
    
    def process_layout(self, components_data, connections_data, root_id=None):
        """
        处理布局
        Args:
            components_data: 元件数据
            connections_data: 连接数据
            root_id: 根节点ID，如果为None则选择第一个元件
        Returns:
            布局结果和布线结果
        """
        # 加载数据
        self.layout_engine.load_components(components_data)
        self.layout_engine.load_connections(connections_data)
        
        # 如果未指定根节点，选择第一个元件
        if root_id is None:
            root_id = components_data[0]['id']
        
        # 生成连接矩阵
        conn_matrix, _ = self.preprocessor.create_connection_matrix(connections_data)
        
        # 压缩特征
        feature = self.preprocessor.compress_connection_matrix(conn_matrix)
        
        # 使用神经网络预测布局参数
        with torch.no_grad():
            layout_params = self.model(feature)
        
        # 使用先验规则生成布局
        layout = self.layout_engine.generate_layout(root_id, layout_params)
        
        # 生成布线
        wiring = self