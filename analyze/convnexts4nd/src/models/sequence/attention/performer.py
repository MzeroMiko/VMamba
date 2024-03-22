# Adapted from https://github.com/HazyResearch/zoo
# in turn adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
"""Implementation of Performer model (https://arxiv.org/abs/2009.14794)."""

import math
import torch
from einops import rearrange, repeat

from fast_transformers.feature_maps.base import FeatureMap

def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block)
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py
def softmax_kernel(data, *, projection_matrix, is_query, softmax_temp=None, eps=1e-4):
    """For key, we expect shape (b, h, s, d) where s is the sequence dimension
    """
    b, h, _, d = data.shape

    if softmax_temp is None:
        softmax_temp = 1 / math.sqrt(d)
    data_normalizer = math.sqrt(softmax_temp)

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)


class PerformerFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        query_dims: int, The input query dimensions in order to sample
                          the noise matrix
        n_features: int, The size of the feature map (should be divisible by 2)
                (default: query_dims)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dims))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: int, Redraw the random matrix every 'redraw' times
                (default: 1)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    def __init__(self, query_dims, n_features=None, ortho_scaling=0, softmax_temp=None,
                 orthogonal=False, redraw=1, deterministic_eval=False):
        super().__init__(query_dims)
        self.n_features = n_features or int(query_dims * math.log(query_dims))
        self.ortho_scaling = ortho_scaling
        # TODO: we're not using @orthogonal atm
        self.orthogonal = orthogonal
        # TODO: we're not using @softmax_temp atm
        self.softmax_temp = 1 / math.sqrt(query_dims) if softmax_temp is None else softmax_temp
        # self.redraw = redraw
        # TODO: not redrawing atm, so I'm setting it to an irrational number
        self.redraw = math.pi
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled projection_matrix
        self.register_buffer("projection_matrix", torch.zeros(self.query_dims, self.n_features))
        self._calls = -1

    def new_feature_map(self, device):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return

        projection_matrix = gaussian_orthogonal_random_matrix(nb_rows=self.n_features,
                                                              nb_columns=self.query_dims,
                                                              scaling=self.ortho_scaling)
        self.register_buffer("projection_matrix", projection_matrix.to(device))

    def forward_queries(self, x):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=True)

    def forward_keys(self, x):
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=False)
