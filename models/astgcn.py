import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.transforms import LaplacianLambdaMax
from dataclasses import dataclass
from typing import List, List, Tuple, Union, Callable, Optional
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian


class ChebConvAttention(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = None,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(ChebConvAttention, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._normalization = normalization
        self._weight = Parameter(torch.Tensor(K, in_channels, out_channels)) 

        if bias:
            self._bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("_bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self._weight)
        if self._bias is not None:
            nn.init.uniform_(self._bias)

    #--forward pass-----
    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight is not None

        return edge_index, edge_weight #for example 307 nodes as deg, 340 edges , 307 nodes as self connections

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        spatial_attention: torch.FloatTensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        
        if self._normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self._normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )
        row, col = edge_index # refer to the index of each note each is a list of nodes not a number # (954, 954)
        Att_norm = norm * spatial_attention[:, row, col] # spatial_attention for example (32, 307, 307), -> (954) * (32, 954) -> (32, 954)
        num_nodes = x.size(self.node_dim) #for example 307
        # (307, 307) * (32, 307, 307) -> (32, 307, 307) -permute-> (32, 307,307) * (32, 307, 1) -> (32, 307, 1)
        TAx_0 = torch.matmul(
            (torch.eye(num_nodes).to(edge_index.device) * spatial_attention).permute(
                0, 2, 1
            ),
            x,
        ) #for example (32, 307, 1)
        out = torch.matmul(TAx_0, self._weight[0]) #for example (32, 307, 1) * [1, 64] -> (32, 307, 64)
        edge_index_transpose = edge_index[[1, 0]]
        if self._weight.size(0) > 1:
            TAx_1 = self.propagate(
                edge_index_transpose, x=TAx_0, norm=Att_norm, size=None
            )
            out = out + torch.matmul(TAx_1, self._weight[1])

        for k in range(2, self._weight.size(0)):
            TAx_2 = self.propagate(edge_index_transpose, x=TAx_1, norm=norm, size=None)
            TAx_2 = 2.0 * TAx_2 - TAx_0
            out = out + torch.matmul(TAx_2, self._weight[k])
            TAx_0, TAx_1 = TAx_1, TAx_2

        if self._bias is not None:
            out += self._bias

        return out #? (b, N, F_out) (32, 307, 64)

    def message(self, x_j, norm):
        if norm.dim() == 1:  # true
            return norm.view(-1, 1) * x_j  # (954, 1) * (32, 954, 1) -> (32, 954, 1)
        else:
            d1, d2 = norm.shape
            return norm.view(d1, d2, 1) * x_j

    def __repr__(self):
        return "{}({}, {}, K={}, normalization={})".format(
            self.__class__.__name__,
            self._in_channels,
            self._out_channels,
            self._weight.size(0),
            self._normalization,
        )

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(SpatialAttention, self).__init__()

        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))  #for example (12)
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps)) #for example (1, 12)
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels)) #for example (1)
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices)) #for example (1,307, 307)
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices)) #for example (307, 307)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        # lhs = left hand side embedding;
        # to calculcate it : 
        # multiply with W1 (B, N, F_in, T)(T) -> (B,N,F_in)
        # multiply with W2 (B,N,F_in)(F_in,T)->(B,N,T)
        # for example (32, 307, 1, 12) * (12) -> (32, 307, 1) * (1, 12) -> (32, 307, 12) 
        LHS = torch.matmul(torch.matmul(X, self._W1), self._W2)
        
        # rhs = right hand side embedding
        # to calculcate it : 
        # mutliple W3 with X (F)(B,N,F,T)->(B, N, T) 
        # transpose  (B, N, T)  -> (B, T, N)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12) -transpose-> (32, 12, 307)
        RHS = torch.matmul(self._W3, X).transpose(-1, -2)
        
        # Then, we multiply LHS with RHS : 
        # (B,N,T)(B,T, N)->(B,N,N)
        # for example (32, 307, 12) * (32, 12, 307) -> (32, 307, 307) 
        # Then multiply Vs(N,N) with the output
        # (N,N)(B, N, N)->(B,N,N) (32, 307, 307)
        # for example (307, 307) *  (32, 307, 307) ->   (32, 307, 307)
        S = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(LHS, RHS) + self._bs))
        S = F.softmax(S, dim=1)
        return S # (B,N,N) for example (32, 307, 307)

class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(TemporalAttention, self).__init__()

        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))  # for example 307
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices)) #for example (1, 307)
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))  # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(1, num_of_timesteps, num_of_timesteps)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))  #for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        # lhs = left hand side embedding;
        # to calculcate it : 
        # permute x:(B, N, F_in, T) -> (B, T, F_in, N)  
        # multiply with U1 (B, T, F_in, N)(N) -> (B,T,F_in)
        # multiply with U2 (B,T,F_in)(F_in,N)->(B,T,N)
        # for example (32, 307, 1, 12) -premute-> (32, 12, 1, 307) * (307) -> (32, 12, 1) * (1, 307) -> (32, 12, 307) 
        LHS = torch.matmul(torch.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2) # (32, 12, 307) 
        
        
        #rhs = right hand side embedding
        # to calculcate it : 
        # mutliple U3 with X (F)(B,N,F,T)->(B, N, T)
        # for example (1)(32, 307, 1, 12) -> (32, 307, 12)
        RHS = torch.matmul(self._U3, X) # (32, 307, 12)
        
        # Them we multiply LHS with RHS : 
        # (B,T,N)(B,N,T)->(B,T,T)
        # for example (32, 12, 307) * (32, 307, 12) -> (32, 12, 12) 
        # Then multiply Ve(T,T) with the output
        # (T,T)(B, T, T)->(B,T,T)
        # for example (12, 12) *  (32, 12, 12) ->   (32, 12, 12)
        E = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(LHS, RHS) + self._be))
        E = F.softmax(E, dim=1) #  (B, T, T)  for example (32, 12, 12)
        return E

class ASTGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        K: int,
        nb_chev_filter: int,
        nb_time_filter: int,
        time_strides: int,
        num_of_vertices: int,
        num_of_timesteps: int,
        normalization: Optional[str] = None,
        bias: bool = True,
    ):
        super(ASTGCNBlock, self).__init__()

        self._temporal_attention = TemporalAttention(
            in_channels, num_of_vertices, num_of_timesteps
        )
        self._spatial_attention = SpatialAttention(
            in_channels, num_of_vertices, num_of_timesteps
        )
        self._chebconv_attention = ChebConvAttention(
            in_channels, nb_chev_filter, K, normalization, bias
        )
        self._time_convolution = nn.Conv2d(
            nb_chev_filter,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self._residual_convolution = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self._normalization = normalization

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> torch.FloatTensor:
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape # (32, 307, 1, 12)

        X_tilde = self._temporal_attention(X) # (b, T, T)  (32, 12, 12) * reshaped x(32, 307, 12)  -reshape> (32, 307, 1, 12)
        # xreshaped is e.g. (32, 307, 12) * (32, 12, 12) -then_reshaped> (32, 307, 1, 12)
        X_tilde = torch.matmul(X.reshape(batch_size, -1, num_of_timesteps), X_tilde)
        X_tilde = X_tilde.reshape(
            batch_size, num_of_vertices, num_of_features, num_of_timesteps
        )
        X_tilde = self._spatial_attention(X_tilde)  # (B,N,N) for example (32, 307, 307)

        if not isinstance(edge_index, list):
            data = Data(
                edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices
            )
            if self._normalization != "sym":
                lambda_max = LaplacianLambdaMax()(data).lambda_max
            else:
                lambda_max = None
            X_hat = []
            for t in range(num_of_timesteps):
                X_hat.append(
                    torch.unsqueeze(
                        self._chebconv_attention(
                            X[:, :, :, t], edge_index, X_tilde, lambda_max=lambda_max
                        ),
                        -1,
                    )
                )

            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        else:
            X_hat = []
            for t in range(num_of_timesteps):
                data = Data(
                    edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices
                )
                if self._normalization != "sym":
                    lambda_max = LaplacianLambdaMax()(data).lambda_max
                else:
                    lambda_max = None
                X_hat.append(
                    torch.unsqueeze(
                        self._chebconv_attention(
                            X[:, :, :, t], edge_index[t], X_tilde, lambda_max=lambda_max
                        ),
                        -1,
                    )
                )
            X_hat = F.relu(torch.cat(X_hat, dim=-1))

        # (b,N,F,T)->(b,F,N,T) for example (32, 307, 64, 12) -premute->(32, 64, 307,12)
        # then convolution along the time axis is applied
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3)) # will give (32, 64, 307,12)
        # (b,N,F,T)-permute>(b,F,N,T) (1,1)->(b,F,N,T)  (32, 64, 307, 12)
        X = self._residual_convolution(X.permute(0, 2, 1, 3))   # will also give (32, 64, 307,12)
        #-adding X + X_hat->(32, 64, 307, 12)-premuting-> (32, 12, 307, 64)-layer_normalization_-premuting->(32, 307, 64,12) 
        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X # (b,N,F,T) for example (32, 307, 64,12) 
    

@dataclass
class ASTGCNConfig:
    nb_block: int = 3
    in_channels: int = 12
    K: int = 3
    nb_chev_filter: int = 12
    nb_time_filter: int = 12
    time_strides: int = 3
    num_for_predict: int = 12
    len_input: int = 12
    num_of_vertices: int = 307
    normalization: Optional[str] = None
    bias: bool = True

class ASTGCN(nn.Module):
    def __init__(
        self,
        config: ASTGCNConfig,
        adjacency: torch.Tensor,
    ):

        super().__init__()
        edge_index, edge_weight = dense_to_sparse(adjacency)
        # edge_weight=torch.unsqueeze(edge_weight,dim=-1)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        self._blocklist = nn.ModuleList(
            [
                ASTGCNBlock(
                    config.in_channels,
                    config.K,
                    config.nb_chev_filter,
                    config.nb_time_filter,
                    config.time_strides,
                    config.num_of_vertices,
                    config.len_input,
                    config.normalization,
                    config.bias,
                )
            ]
        )

        self._blocklist.extend(
            [
                ASTGCNBlock(
                    config.nb_time_filter,
                    config.K,
                    config.nb_chev_filter,
                    config.nb_time_filter,
                    1,
                    config.num_of_vertices,
                    config.len_input // config.time_strides,
                    config.normalization,
                    config.bias,
                )
                for _ in range(config.nb_block - 1)
            ]
        )

        self._final_conv = nn.Conv2d(
            int(config.len_input / config.time_strides),
            config.num_for_predict,
            kernel_size=(1, config.nb_time_filter),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the parameters.
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

    def forward(
        self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        x=x.permute(0,2,1,3)
        edge_index, _ = self._resolve_graph(adjacency)
        for block in self._blocklist:
            # original x is (B,N,F_in,T) will give (B,N,F_out,T) for example (32, 307, 1, 12) -> (32, 307, 64, 12) 
            x = block(x, edge_index) 

        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1) 
        # for example (32, 307, 64, 12) -permute-> (32, 12, 307,64) -final_conv-> (32, 12, 307, 1)
        x = self._final_conv(x.permute(0, 3, 1, 2))
        # (b,c_out*T,N)->(b,N,T)
        x = x[:, :, :, -1] # (b,c_out*T,N) for example (32, 12, 307)
        x = x.permute(0, 2, 1) # (b,T,N)-> (b,N,T)
        return x #(b,N,T) for exmaple (32, 307,12)