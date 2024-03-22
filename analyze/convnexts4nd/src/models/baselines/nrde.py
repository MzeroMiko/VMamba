"""Neural Rough Differential Equations."""

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import bisect


def rdeint(logsig, h0, func, method='rk4', adjoint=False, return_sequences=False):
    """Analogous to odeint but for RDEs.
    Note that we do not have time intervals here. This is because the log-ode method is always evaluated on [0, 1] and
    thus are grid is always [0, 1, ..., num_intervals+1].
    Args:
        logsig (torch.Tensor): A tensor of logsignature of shape [N, L, logsig_dim]
        h0 (torch.Tensor): The initial value of the hidden state.
        func (nn.Module): The function to apply to the state h0.
        method (str): The solver to use.
        adjoint (bool): Set True to use the adjoint method.
        return_sequences (bool): Set True to return a prediction at each step, else return just terminal time.
    Returns:
        torch.Tensor: The values of the hidden states at the specified times. This has shape [N, L, num_hidden].
    """
    # Method to get the logsig value
    logsig_getter = _GetLogsignature(logsig)

    # A cell to apply the output of the function linearly to correct log-signature piece.
    cell = _NRDECell(logsig_getter, func)

    # Set options
    t, options, = set_options(logsig, return_sequences=return_sequences)

    # Solve
    odeint_func = odeint_adjoint if adjoint else odeint
    output = odeint_func(func=cell, y0=h0, t=t, method=method, options=options).transpose(0, 1)

    return output


def set_options(logsig, return_sequences=False, eps=1e-5):
    """Sets the options to be passed to the relevant `odeint` function.
    Args:
        logsig (torch.Tensor): The logsignature of the path.
        return_sequences (bool): Set True if a regression problem where we need the full sequence. This requires us
            specifying the time grid as `torch.arange(0, T_final)` which is less memory efficient that specifying
            the times `t = torch.Tensor([0, T_final])` along with an `step_size=1` in the options.
        eps (float): The epsilon perturbation to make to integration points to distinguish the ends.
    Returns:
        torch.Tensor, dict: The integration times and the options dictionary.
    """
    length = logsig.size(1) + 1
    if return_sequences:
        t = torch.arange(0, length, dtype=torch.float).to(logsig.device)
        options = {'eps': eps}
    else:
        options = {'step_size': 1, 'eps': eps}
        t = torch.Tensor([0, length]).to(logsig.device)
    return t, options


class _GetLogsignature:
    """Given a time value, gets the corresponding piece of the log-signature.
    When performing a forward solve, torchdiffeq will give us the time value that it is solving the ODE on, and we need
    to return the correct piece of the log-signature corresponding to that value. For example, let our intervals ends
    be the integers from 0 to 10. Then if the time value returned by torchdiffeq is 5.5, we need to return the
    logsignature on [5, 6]. This function simply holds the logsignature, and interval end times, and returns the
    correct logsignature given any time.
    """
    def __init__(self, logsig):
        self.knots = range(logsig.size(1))
        self.logsig = logsig

    def __getitem__(self, t):
        index = bisect.bisect(self.knots, t) - 1
        return self.logsig[:, index]


class _NRDECell(nn.Module):
    """Applies the function to the previous hidden state, and then applies the output linearly onto the log-signature.
    The NeuralRDE model solves the following equation:
        dH = f(H) o logsignature(X_{t_i, t_{i+1}) dt;    H(0) = H_t_i.
    given a function f, this class applies that function to the hidden state, and then applies that result linearly onto
    the correct piece of the logsignature.
    """
    def __init__(self, logsig_getter, func):
        super().__init__()
        self.logsig_getter = logsig_getter
        self.func = func

    def forward(self, t, h):
        A = self.func(h)
        output = torch.bmm(A, self.logsig_getter[t].unsqueeze(2)).squeeze(2)
        return output


class NeuralRDE(nn.Module):
    """The generic module for learning with Neural RDEs.
    This class wraps the `NeuralRDECell` that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `NeuralRDECell` as the function that
    computes the update.
    Here we model the dynamics of some abstract hidden state H via a CDE, and the response as a linear functional of the
    hidden state, that is:
        dH = f(H)dX;    Y = L(H).
    """
    def __init__(self,
                 initial_dim,
                 logsig_dim,
                 hidden_dim,
                 output_dim,
                 hidden_hidden_dim=15,
                 num_layers=3,
                 apply_final_linear=True,
                 solver='midpoint',
                 adjoint=False,
                 return_sequences=False):
        """
        Args:
            initial_dim (int): We use the initial value (t_0 x_0) as an initial condition else we have translation
                invariance.
            logsig_dim (int): The dimension of the log-signature.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
            apply_final_linear (bool): Set False to ignore the final linear output.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            adjoint (bool): Set True to use odeint_adjoint.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
        """
        super().__init__()
        self.initial_dim = initial_dim
        self.logsig_dim = logsig_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.apply_final_linear = apply_final_linear
        self.solver = solver
        self.adjoint = adjoint
        self.return_sequences = return_sequences

        # Initial to hidden
        self.initial_linear = nn.Linear(initial_dim, hidden_dim)

        # The net applied to h_prev
        self.func = _NRDEFunc(hidden_dim, logsig_dim, hidden_dim=hidden_hidden_dim, num_layers=num_layers)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim) if apply_final_linear else lambda x: x

    def forward(self, inputs):
        # Setup the inital hidden layer
        assert len(inputs) == 2, "`inputs` must be a 2-tuple containing `(inital_values, logsig)`."
        initial, logsig = inputs
        h0 = self.initial_linear(initial)

        # Perform the adjoint operation
        out = rdeint(
            logsig, h0, self.func, method=self.solver, adjoint=self.adjoint, return_sequences=self.return_sequences
        )

        # Outputs
        outputs = self.final_linear(out[:, -1, :]) if not self.return_sequences else self.final_linear(out)

        return outputs


class _NRDEFunc(nn.Module):
    """The function applied to the hidden state in the log-ode method.
    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) o logsig(X_{[t_i, t_{i+1}]})
    To build a custom version, simply use any NN architecture such that `input_dim` is the size of the hidden state,
    and the output dim must be of size `input_dim * logsig_dim`. Simply reshape the output onto a tensor of size
    `[batch, input_dim, logsig_dim]`.
    """
    def __init__(self, input_dim, logsig_dim, num_layers=1, hidden_dim=15):
        super().__init__()
        self.input_dim = input_dim
        self.logsig_dim = logsig_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Additional layers are just hidden to hidden with relu activation
        additional_layers = [nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 1) if num_layers > 1 else []

        # The net applied to h_prev
        self.net = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim),
            *additional_layers,
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim * logsig_dim),
        ]) if num_layers > 0 else nn.Linear(input_dim, input_dim * logsig_dim)

    def forward(self, h):
        return self.net(h).view(-1, self.input_dim, self.logsig_dim)
