"""Wrapper around expRNN's Orthogonal class for convenience."""

from .exprnn.orthogonal import Orthogonal
from .exprnn.trivializations import expm, cayley_map
from .exprnn.initialization import henaff_init_, cayley_init_

param_name_to_param = {'cayley': cayley_map, 'expm': expm}
init_name_to_init = {'henaff': henaff_init_, 'cayley': cayley_init_}

class OrthogonalLinear(Orthogonal):
    def __init__(self, d_input, d_output, method='dtriv', init='cayley', K=100):
        """Wrapper around expRNN's Orthogonal class taking care of parameter names."""
        if method == "exprnn":
            mode = "static"
            param = 'expm'
        elif method == "dtriv":
            # We use 100 as the default to project back to the manifold.
            # This parameter does not really affect the convergence of the algorithms, even for K=1
            mode = ("dynamic", K, 100) # TODO maybe K=30? check exprnn codebase
            param = 'expm'
        elif method == "cayley":
            mode = "static"
            param = 'cayley'
        else:
            assert False, f"OrthogonalLinear: orthogonal method {method} not supported"

        param = param_name_to_param[param]
        init_A = init_name_to_init[init]
        super().__init__(d_input, d_output, init_A, mode, param)

        # Scale LR by factor of 10
        self.A._lr_scale = 0.1
