import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNWithMLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, gcn_out_dim=256, num_classes=2, num_layers=6):
        super(GCNWithMLP, self).__init__()

        # GCN
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(nn.Linear(in_dim, hidden_dim))  # 第1层
        for _ in range(num_layers - 2):  # 中间层
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.gcn_layers.append(nn.Linear(hidden_dim, gcn_out_dim))  # 第6层输出为 gcn_out_dim

        # MLP
        self.mlp_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(gcn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        for layer in self.gcn_layers:
            x = F.relu(layer(x))
        x = self.mlp_head(x)
        return x

# parameter cal
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# main for test
def main():
    model = GCNWithMLP()
    input_img = torch.randn(4, 2048)  # T2WI + ADC + T2WI_ROIs + ADC_ ROIs
    output_feature = model(input_img)

    total_params, trainable_params = count_parameters(model)
    print(f"total parameters: {total_params:,}")
    print(f"parameters for training: {trainable_params:,}")

if __name__ == "__main__":
    main()