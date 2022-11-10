import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss

class NeuralCollaborativeFiltering:

    def __init__(self, args, data, load=False):
        super().__init__()

        # self.criterion = RMSELoss()
        # self.train_dataloader = data['train_dataloader']
        # self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx=np.array((1, ), dtype=np.long)

        self.embed_dim = args.NCF_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")#args.DEVICE

        self.mlp_dims = args.NCF_MLP_DIMS
        self.dropout = args.NCF_DROPOUT

        self.model = _NeuralCollaborativeFiltering(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        if load:
            self.model.load_state_dict(torch.load('NCF_epoch_1_rmse_2.2396299296441464.pth', map_location=self.device))

    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields, embeddings in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                # fields = fields[0].to(self.device)
                fields, embeddings = fields.to(self.device), embeddings.to(self.device)
                y = self.model(fields, embeddings)
                predicts.extend(y.tolist())
        return predicts

class _NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + 512 + embed_dim, 1)
        self.linear1 = torch.nn.Linear(512, 2048)
        self.linear2 = torch.nn.Linear(2048, 4096)
        self.linear3 = torch.nn.Linear(4096, 2048)
        self.linear4 = torch.nn.Linear(2048, 1024)
        self.linear5 = torch.nn.Linear(1024, 512)
        self.bn1 = torch.nn.BatchNorm1d(2048)
        self.bn2 = torch.nn.BatchNorm1d(4096)
        self.bn3 = torch.nn.BatchNorm1d(2048)
        self.bn4 = torch.nn.BatchNorm1d(1024)
        self.bn5 = torch.nn.BatchNorm1d(512)
        relu = torch.nn.ReLU()
        dropout = torch.nn.Dropout()
        self.emb_model = torch.nn.Sequential(self.linear1, relu, self.bn1, dropout,
                                             self.linear2, relu, self.bn2, dropout,
                                             self.linear3, relu, self.bn3, dropout,
                                             self.linear4, relu, self.bn4, dropout,
                                             self.linear5, relu, self.bn5, dropout,)
        
    def forward(self, x, embeddings):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        # import pdb; pdb.set_trace();
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))

        embeddings = self.emb_model(embeddings)

        x = torch.cat([gmf, x, embeddings], dim=1)
        x = self.fc(x).squeeze(1)
        return x

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)