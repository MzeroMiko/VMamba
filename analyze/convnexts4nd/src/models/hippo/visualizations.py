"""Standalone implementation of HiPPO operators.

Contains experiments for the function reconstruction experiment in original HiPPO paper,
as well as new animations from "How to Train Your HiPPO".

This file ports the notebook notebooks/hippo_function_approximation.ipynb,
which is recommended if Jupyter is supported.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from einops import rearrange, repeat, reduce

import src.models.functional.unroll as unroll # Not necessary, can comment out and set fast=False in HiPPO modules

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns
sns.set(rc={
    "figure.dpi":300,
    'savefig.dpi':300,
    'animation.html':'jshtml',
    'animation.embed_limit':100, # Max animation size in Mb
})
# sns.set_context('notebook')
sns.set_style('ticks') # or 'whitegrid'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# HiPPO matrices
def transition(measure, N, **measure_args):
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([np.zeros(N//2), freqs], axis=-1).reshape(-1)[1:]
        A = 2*np.pi*(-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2
        B[0] = 2**.5
        A = A - B[:, None] * B[None, :]
        # A = A - np.eye(N)
        B *= 2**.5
        B = B[:, None]

    return A, B

def measure(method, c=0.0):
    if method == 'legt':
        fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    elif method == 'legs':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method == 'lagt':
        fn = lambda x: np.heaviside(x, 1.0) * np.exp(-x)
    elif method in ['fourier']:
        fn = lambda x: np.heaviside(x, 1.0) * np.heaviside(1.0-x, 1.0)
    else: raise NotImplementedError
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted

def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
    returns: shape (T, N) where T is length of vals
    """
    if method == 'legt':
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 2*vals-1).T
        eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
    elif method == 'legs':
        _vals = np.exp(-vals)
        eval_matrix = ss.eval_legendre(np.arange(N)[:, None], 1-2*_vals).T # (L, N)
        eval_matrix *= (2*np.arange(N)+1)**.5 * (-1)**np.arange(N)
    elif method == 'lagt':
        vals = vals[::-1]
        eval_matrix = ss.eval_genlaguerre(np.arange(N)[:, None], 0, vals)
        eval_matrix = eval_matrix * np.exp(-vals / 2)
        eval_matrix = eval_matrix.T
    elif method == 'fourier':
        cos = 2**.5 * np.cos(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
        sin = 2**.5 * np.sin(2*np.pi*np.arange(N//2)[:, None]*(vals)) # (N/2, T/dt)
        cos[0] /= 2**.5
        eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N) # (T/dt, N)
#     print("eval_matrix shape", eval_matrix.shape)

    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0

    p = torch.tensor(eval_matrix)
    p *= np.exp(-c*vals)[:, None] # [::-1, None]
    return p


class HiPPOScale(nn.Module):
    """Vanilla HiPPO-LegS model (scale invariant instead of time invariant)."""

    def __init__(self, N, method='legs', max_length=1024, discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(method, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked)) # (max_length, N, N)
        self.register_buffer('B_stacked', torch.Tensor(B_stacked)) # (max_length, N)

        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor((B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T  )

    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
            return result

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for t, f in enumerate(inputs):
            c = F.linear(c, self.A_stacked[t]) + self.B_stacked[t] * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = self.eval_matrix.to(c) @ c.unsqueeze(-1)
        return a

class HiPPO(nn.Module):
    """Linear time invariant x' = Ax + Bu."""

    def __init__(self, N, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """

        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c

        A, B = transition(method, N)
        A = A + np.eye(N)*c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)

        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA)) # (N, N)
        self.register_buffer('dB', torch.Tensor(dB)) # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c) # (T/dt, N)
        self.measure = measure(self.method)(self.vals)


    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB # (length, ..., N)

        if fast:
            dA = repeat(self.dA, 'm n -> l m n', l=u.size(0))
            return unroll.variable_unroll_matrix(dA, u)

        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)



    def reconstruct(self, c, evals=None): # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        m = self.measure[self.measure != 0.0]

        c = c.unsqueeze(-1)
        y = eval_matrix.to(c) @ c
        return y.squeeze(-1).flip(-1)


### Synthetic data generation

def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",)

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1-set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.: coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal


def plot(T, dt, N, freq):
    np.random.seed(0)
    vals = np.arange(0.0, T, dt)

    u = whitesignal(T, dt, freq=freq)
    u = torch.tensor(u, dtype=torch.float)
    u = u.to(device)

    plt.figure(figsize=(16, 8))
    offset = 0.0
    plt.plot(vals, u.cpu()+offset, 'k', linewidth=1.0)

    # Linear Time Invariant (LTI) methods x' = Ax + Bu
    lti_methods = [
        'legs',
        'legt',
        'fourier',
    ]

    for method in lti_methods:
        hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method)

    # Original HiPPO-LegS, which uses time-varying SSM x' = 1/t [ Ax + Bu]
    # we call this "linear scale invariant"
    lsi_methods = ['legs']
    for method in lsi_methods:
        hippo = HiPPOScale(N=N, method=method, max_length=int(T/dt)).to(device)
        u_hippo = hippo.reconstruct(hippo(u))[-1].cpu()
        plt.plot(vals[-len(u_hippo):], u_hippo, label=method+' (scaled)')


    # plt.xlabel('Time (normalized)', labelpad=-10)
    plt.legend()
    plt.savefig(f'function_approximation.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


# Animation code from HTTYH

def plt_lines(x, y, color, size, label=None):
    return plt.plot(x, y, color, linewidth=size, label=label)[0]

def update_lines(ln, x, y):
    ln.set_data(x, y)

def animate_hippo(
    method,
    T=5, dt=5e-4, N=64, freq=20.0,
    interval=100,
    plot_hippo=False, hippo_offset=0.0, label_hippo=False,
    plot_measure=False, measure_offset=-3.0, label_measure=False,
    plot_coeff=None, coeff_offset=3.0,
    plot_s4=False, s4_offset=6.0,
    plot_hippo_type='line', plot_measure_type='line', plot_coeff_type='line',
    size=1.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=False,
    animate_u=False,
    seed=2,
):
    np.random.seed(seed)

    vals = np.arange(0, int(T/dt)+1)
    L = int(T/dt)+1

    u = torch.FloatTensor(whitesignal(T, dt, freq=freq))
    u = F.pad(u, (1, 0))
    u = u + torch.FloatTensor(np.sin(1.5*np.pi/T*np.arange(0, T+dt, dt))) # add 3/4 of a sin cycle
    u = u.to(device)

    hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
    coef_hippo = hippo(u).cpu().numpy()
    h_hippo = hippo.reconstruct(hippo(u)).cpu().numpy()
    u = u.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))

    if animate_u:
        ln_u = plt_lines([], [], 'k', size, label='Input $u(t)$')
    else:
        plt_lines(vals, u, 'k', size, label='Input $u(t)$')

    if plot_hippo:
        label_args = {'label': 'HiPPO reconstruction'} if label_hippo else {}
        ln = plt_lines([], [], size=size, color='red', **label_args)

    if plot_measure:
        label_args = {'label': 'HiPPO Measure'} if label_measure else {}
        ln_measure = plt_lines(vals, np.zeros(len(vals))+measure_offset, size=size, color='green', **label_args)

    if plot_coeff is None: plot_coeff = []
    if isinstance(plot_coeff, int): plot_coeff = [plot_coeff]
    if len(plot_coeff) > 0:
        ln_coeffs = [
            plt_lines([], [], size=size, color='blue')
            for _ in plot_coeff
        ]
        plt_lines([], [], size=size, color='blue', label='State $x(t)$') # For the legend


    ### Y AXIS LIMITS
    if plot_measure:
        min_y = measure_offset
    else:
        min_y = np.min(u)

    if len(plot_coeff) > 0:
        max_u = np.max(u) + coeff_offset
    else:
        max_u = np.max(u)


    C = np.random.random(N)
    s4 = np.sum(coef_hippo * C, axis=-1)
    max_s4 = 0.0
    if plot_s4:
        ln_s4 = plt_lines([], [], size=size, color='red', label='Output $y(t)$')
        max_s4 = np.max(s4)+s4_offset

    if plot_vline:
        ln_vline = ax.axvline(0, ls='-', color='k', lw=1)

    if plot_legend:
        plt.legend(loc='upper left', fontsize='x-small')


    def init():
        left_endpoint = vals[0]
        ax.set_xlim(left_endpoint, vals[-1]+1)
        ax.set_ylim(min_y, max(max_u, max_s4))
        ax.set_yticks([])
        if not plot_xticks: ax.set_xticks([])
        if not plot_box: plt.box(False)
        return [] # ln,

    def update(frame):
        if animate_u:
            xdata = np.arange(frame)
            ydata = u[:frame]
            update_lines(ln_u, xdata, ydata)

        m = np.zeros(len(vals))
        m[:frame] = hippo.measure_fn(np.arange(frame)*dt)[::-1]
        xdata = vals
        if plot_measure:
            update_lines(ln_measure, xdata, m+measure_offset)

        if plot_hippo:
            ydata = h_hippo[frame] + hippo_offset
            m2 = hippo.measure_fn(np.arange(len(ydata))*dt)[::-1]
            # Remove reconstruction where measure is 0
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame-len(ydata), frame)
            update_lines(ln, xdata, ydata)

        if len(plot_coeff) > 0:
            for coeff, ln_coeff in zip(plot_coeff, ln_coeffs):
                update_lines(ln_coeff, np.arange(frame), coef_hippo[:frame, coeff] + coeff_offset)
        if plot_s4: # Only scale case; scale case should copy plot_hippo logic
            update_lines(ln_s4, np.arange(0, frame), s4[:frame] + s4_offset)

        if plot_vline:
            ln_vline.set_xdata([frame, frame])

        return []

    ani = FuncAnimation(fig, update,
                        frames=np.arange(0, int(T*1000/interval)+1)*int(interval/1000/dt),
                        interval=interval,
                        init_func=init, blit=True)

    return ani


if __name__ == '__main__':
    plot(T=3, dt=1e-3, N=64, freq=3.0)

    # Visualize HiPPO online reconstruction

    ani = animate_hippo(
        'legs', # Try 'legt' or 'fourier'
        T=5, dt=5e-4, N=64, interval=100,
        # T=1, dt=1e-3, N=64, interval=200, # Faster rendering for testing
        size=1.0,

        animate_u=True,
        plot_hippo=True, hippo_offset=0.0, label_hippo=True,
        plot_s4=False, s4_offset=6.0,
        plot_measure=True, measure_offset=-3.0, label_measure=True,
        plot_coeff=[], coeff_offset=3.0,
        plot_legend=True, plot_xticks=True, plot_box=True,
        plot_vline=True,
    )
    ani.save('hippo_legs.gif')

    # Visualize S4

    ani = animate_hippo(
        'legs', # Try 'legt' or 'fourier'
        T=5, dt=5e-4, N=64, interval=100,
        size=1.0,

        animate_u=True,
        plot_hippo=False, hippo_offset=0.0, label_hippo=True,
        plot_s4=True, s4_offset=6.0,
        plot_measure=False, measure_offset=-3.0, label_measure=True,
        plot_coeff=[0,1,2,3], coeff_offset=3.0,
        plot_legend=True, plot_xticks=True, plot_box=True,
        plot_vline=True,
    )
    ani.save('s4_legs.gif')
