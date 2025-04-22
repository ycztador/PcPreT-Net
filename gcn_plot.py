import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'


# 4 MRI features
features_type1 = np.load('features_T2WI.npy')
features_type2 = np.load('features_T2WI_ROIs.npy')
features_type3 = np.load('features_ADC.npy')
features_type4 = np.load('features_ADC_ROIs.npy')

features = np.vstack((features_type1, features_type2, features_type3, features_type4))
num_nodes_type1 = features_type1.shape[0]
num_nodes_type2 = features_type2.shape[0]
num_nodes_type3 = features_type3.shape[0]
num_nodes_type4 = features_type4.shape[0]
num_nodes = features.shape[0] 

# L1-distance
distance_matrix = np.abs(features[:, np.newaxis] - features[np.newaxis, :]).sum(axis=-1)

mean_distance = np.mean(distance_matrix)
std_distance = np.std(distance_matrix)

threshold = mean_distance + 0.5 * std_distance

adjacency_matrix = (distance_matrix < threshold).astype(float)

G = nx.Graph()

for i in range(num_nodes):
    G.add_node(i)

edges = []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes): 
        if adjacency_matrix[i, j] > 0:
            G.add_edge(i, j)
            edges.append((i, j))

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G) 


nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.01, edge_color='gray')

node_colors = ['skyblue'] * num_nodes 
node_colors[:num_nodes_type1] = ['red'] * num_nodes_type1  
node_colors[num_nodes_type1:num_nodes_type1 + num_nodes_type2] = ['yellow'] * num_nodes_type2  
node_colors[num_nodes_type1 + num_nodes_type2:num_nodes_type1 + num_nodes_type2 + num_nodes_type3] = ['green'] * num_nodes_type3 
node_colors[num_nodes_type1 + num_nodes_type2 + num_nodes_type3:] = ['steelblue'] * num_nodes_type4 

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)


legend_labels = ['T2WI Slice', 'T2WI ROI', 'ADC Slice', 'ADC ROI']
legend_colors = ['red', 'yellow', 'green', 'steelblue']
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
plt.legend(handles=patches, loc='upper right')

plt.title('Graph Structure with ResNet50 Features Extractor')
plt.axis('off') 
plt.show()
