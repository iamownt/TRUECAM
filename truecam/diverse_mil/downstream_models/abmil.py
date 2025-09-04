import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMIL, self).__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 2)

    def forward(self, H):
        tanh_VH = torch.tanh(self.V(H))
        attention_scores = self.w(tanh_VH)
        attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights


class GatedAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_input, dropout_hidden):
        super(GatedAttention, self).__init__()
        assert 0 <= dropout_input <= 1 and 0 <= dropout_hidden <= 1
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_input))
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_hidden))
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        c = a.mul(b)
        c = self.w(c)
        prob = F.softmax(c, dim=1)  # abmil likes to use batch size 1
        return (prob * x).sum(dim=1)


# class GatedABMIL(nn.Module):
#     def __init__(self, embed_dim: int = 1024, hdim1: int = 512, hdim2: int = 384, n_classes: int = 2):
#         super(GatedABMIL, self).__init__()
#         self.feature_extractor = nn.Sequential(nn.Linear(embed_dim, hdim1), nn.ReLU(), nn.Dropout(0.1))
#         self.attention_layer = GatedAttention(hdim1, hdim2, dropout_input=0.25, dropout_hidden=0.25)
#         self.classifier = nn.Linear(hdim1, n_classes)
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#     def forward(self, x, **kwargs):
#         x = self.feature_extractor(x)
#         x = self.attention_layer(x)
#         return self.classifier(x, **kwargs)

class GatedABMIL(nn.Module):
    """https://github.com/mahmoodlab/CLAM/models/model_clam.py
       The only differences is that we use single mapping to enable uni and conch with the same hidden state for attention
    """

    def __init__(self, embed_dim: int = 1024, hdim1: int = 512, hdim2: int = 384, n_classes: int = 2):
        super(GatedABMIL, self).__init__()
        # if embed_dim == 512:
        #     self.fair_proj = nn.Linear(embed_dim, 1024)
        #     print("use fair projection")
        #     embed_dim = 1024
        # else:
        #     self.fair_proj = nn.Identity()
        self.feature_extractor = nn.Sequential(nn.Linear(embed_dim, hdim1), nn.ReLU(), nn.Dropout(0.1))
        self.attention_layer = GatedAttention(hdim1, hdim2, dropout_input=0.25, dropout_hidden=0.25)
        self.classifier = nn.Linear(hdim1, n_classes)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, **kwargs):
        # x = self.fair_proj(x)
        x = self.feature_extractor(x)
        x = self.attention_layer(x)
        return self.classifier(x, **kwargs)


if __name__ == "__main__":
    spec_norm_bound = 2.0
    spec_norm_replace_list = ["Linear", "Conv2D"]
    sample_input = torch.randn(1, 1000, 512)
    model = GatedABMIL(embed_dim=512, n_classes=3)
    model = convert_to_sn_my(model, spec_norm_replace_list, spec_norm_bound)
    print(model)
    output = model(sample_input)
    print("Output shape:", output.shape)