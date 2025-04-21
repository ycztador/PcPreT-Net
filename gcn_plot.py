import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'


# 加载四种类型的 ROI 特征
features_type1 = np.load('features_T2WI.npy')
features_type2 = np.load('features_T2WI_ROIs.npy')
features_type3 = np.load('features_ADC.npy')
features_type4 = np.load('features_ADC_ROIs.npy')
# 合并特征并计算节点数量
features = np.vstack((features_type1, features_type2, features_type3, features_type4))
num_nodes_type1 = features_type1.shape[0]
num_nodes_type2 = features_type2.shape[0]
num_nodes_type3 = features_type3.shape[0]
num_nodes_type4 = features_type4.shape[0]
num_nodes = features.shape[0]  # 总节点数量

# 计算 L1 距离矩阵
distance_matrix = np.abs(features[:, np.newaxis] - features[np.newaxis, :]).sum(axis=-1)
# # 计算 L2 距离矩阵
# distance_matrix = np.linalg.norm(features[:, np.newaxis] - features[np.newaxis, :], axis=-1)

# 计算均值和标准差
mean_distance = np.mean(distance_matrix)
std_distance = np.std(distance_matrix)

# 设置阈值为 a + 0.5 * b
threshold = mean_distance + 0.5 * std_distance

# 构建邻接矩阵
adjacency_matrix = (distance_matrix < threshold).astype(float)

# 创建图
G = nx.Graph()

# 添加节点
for i in range(num_nodes):
    G.add_node(i)

# 添加边，并记录无连接边
edges = []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):  # 避免重复添加
        if adjacency_matrix[i, j] > 0:
            G.add_edge(i, j)
            edges.append((i, j))

# 绘制图
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # 使用弹簧布局

# 绘制有连接的边，设置透明度
nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.01, edge_color='gray')

# 绘制节点
node_colors = ['skyblue'] * num_nodes  # 默认颜色
node_colors[:num_nodes_type1] = ['red'] * num_nodes_type1  # 第一种类型颜色
node_colors[num_nodes_type1:num_nodes_type1 + num_nodes_type2] = ['yellow'] * num_nodes_type2  # 第二种类型颜色
node_colors[num_nodes_type1 + num_nodes_type2:num_nodes_type1 + num_nodes_type2 + num_nodes_type3] = ['green'] * num_nodes_type3  # 第三种类型颜色
node_colors[num_nodes_type1 + num_nodes_type2 + num_nodes_type3:] = ['steelblue'] * num_nodes_type4  # 第四种类型颜色

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)

# 创建图例
legend_labels = ['T2WI Slice', 'T2WI ROI', 'ADC Slice', 'ADC ROI']
legend_colors = ['red', 'yellow', 'green', 'steelblue']
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
plt.legend(handles=patches, loc='upper right')

plt.title('Graph Structure with ResNet50 Features Extractor')
plt.axis('off')  # 关闭坐标轴
# plt.savefig('使用ResNey50的图网络结构.pdf')
plt.show()
