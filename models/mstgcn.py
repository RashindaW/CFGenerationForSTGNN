import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.transforms import LaplacianLambdaMax
from dataclasses import dataclass
import math
from typing import Tuple, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSTGCNBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
    ):
        super(MSTGCNBlock, self).__init__()

        self._cheb_conv = ChebConv(in_channels, nb_chev_filter, K, normalization=None)

        self._time_conv = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )

        self._residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )

        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:

        batch_size,num_of_vertices,in_channels,num_of_timesteps = X.shape
        # X=X.permute(0,2,1,3)

        if not isinstance(edge_index, list):

            lambda_max = LaplacianLambdaMax()(
                Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_of_vertices)
            ).lambda_max

            X_tilde = X.permute(2, 0, 1, 3)
            X_tilde = X_tilde.reshape(
                num_of_vertices, in_channels, num_of_timesteps * batch_size
            )
            X_tilde = X_tilde.permute(2, 0, 1)
            # print(X_tilde.shape,edge_index.shape,edge_weight.shape,lambda_max)
            X_tilde = F.relu(
                self._cheb_conv(x=X_tilde, edge_index=edge_index, edge_weight=edge_weight, lambda_max=lambda_max)
            )
            X_tilde = X_tilde.permute(1, 2, 0)
            X_tilde = X_tilde.reshape(
                num_of_vertices, self.nb_time_filter, batch_size, num_of_timesteps
            )
            X_tilde = X_tilde.permute(2, 0, 1, 3)

        else:
            X_tilde = []
            for t in range(num_of_timesteps):
                lambda_max = LaplacianLambdaMax()(
                    Data(
                        edge_index=edge_index[t],
                        edge_attr=edge_weight[t] if edge_weight is not None else None,
                        num_nodes=num_of_vertices,
                    )
                ).lambda_max
                X_tilde.append(
                    torch.unsqueeze(
                        self._cheb_conv(
                            X[:, :, :, t], edge_index[t], lambda_max=lambda_max
                        ),
                        -1,
                    )
                )
            X_tilde = F.relu(torch.cat(X_tilde, dim=-1))

        X_tilde = self._time_conv(X_tilde.permute(0, 2, 1, 3))
        X = self._residual_conv(X.permute(0, 2, 1, 3))
        X = self._layer_norm(F.relu(X + X_tilde).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X
    

@dataclass
class MSTGCNConfig:
    nb_block: int=12
    in_channels: int=12
    K: int=3
    nb_chev_filter: int=12
    nb_time_filter: int=12
    time_strides: int=1
    num_for_predict: int=1
    len_input: int=1


class MSTGCN(nn.Module):
    def __init__(
        self,
        config: MSTGCNConfig, adjacency: torch.Tensor
    ):
        super().__init__()
        edge_index, edge_weight = dense_to_sparse(adjacency)
        # edge_weight=torch.unsqueeze(edge_weight,dim=-1)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self._blocklist = nn.ModuleList(
            [MSTGCNBlock(config.in_channels, config.K, config.nb_chev_filter, config.nb_time_filter, config.time_strides)]
        )

        self._blocklist.extend(
            [
                MSTGCNBlock(config.nb_time_filter, config.K, config.nb_chev_filter, config.nb_time_filter, 1)
                for _ in range(config.nb_block - 1)
            ]
        )

        self._final_conv = nn.Conv2d(
            int(config.len_input / config.time_strides),
            config.num_for_predict,
            kernel_size=(1, config.nb_time_filter),
        )
        # self.linear = nn.Linear(config.hidden_channels, config.horizon)
        self._reset_parameters()
        
    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _resolve_graph(self, adjacency: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if adjacency is None:
            return self.edge_index, self.edge_weight
        edge_index, edge_weight = dense_to_sparse(adjacency)
        # edge_weight=torch.unsqueeze(edge_weight,dim=-1)
        return edge_index.to(self.edge_index.device), edge_weight.to(self.edge_weight.device)

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        edge_index, edge_weight = self._resolve_graph(adjacency)
        x=x.permute(0,2,1,3) #mstgcn requires (batch, nodes, features, steps)
        for block in self._blocklist:
            x = block(x, edge_index,edge_weight)

        x = self._final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return x
