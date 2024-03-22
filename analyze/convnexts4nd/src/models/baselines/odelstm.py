# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
"""Adapted from ODE-LSTM https://github.com/mlech26l/ode-lstms/."""

import torch
import torch.nn as nn
from torchdyn.models import NeuralDE
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class ODELSTMCell(nn.Module):
    def __init__(self, d_model, d_hidden, solver_type="dopri5"):
        super(ODELSTMCell, self).__init__()
        self.solver_type = solver_type
        self.fixed_step_solver = solver_type.startswith("fixed_")
        self.lstm = nn.LSTMCell(d_model, d_hidden)
        # 1 hidden layer NODE
        self.f_node = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.d_model = d_model
        self.d_hidden = d_hidden
        if not self.fixed_step_solver:
            self.node = NeuralDE(self.f_node, solver=solver_type)
        else:
            options = {
                "fixed_euler": self.euler,
                "fixed_heun": self.heun,
                "fixed_rk4": self.rk4,
            }
            if not solver_type in options.keys():
                raise ValueError("Unknown solver type '{:}'".format(solver_type))
            self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        new_h, new_c = self.lstm(input, hx)
        if self.fixed_step_solver:
            new_h = self.solve_fixed(new_h, ts)
        else:
            indices = torch.argsort(ts)
            batch_size = ts.size(0)
            device = input.device
            s_sort = ts[indices]
            s_sort = s_sort + torch.linspace(0, 1e-4, batch_size, device=device)
            # HACK: Make sure no two points are equal
            trajectory = self.node.trajectory(new_h, s_sort)
            new_h = trajectory[indices, torch.arange(batch_size, device=device)]

        return (new_h, new_c)

    def solve_fixed(self, x, ts):
        ts = ts.view(-1, 1)
        for i in range(3):  # 3 unfolds
            x = self.node(x, ts * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)

        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class ODELSTM(nn.Module):
    def __init__(
        self,
        d_model,
        d_output=None,
        d_hidden=None,
        return_sequences=True,
        solver_type="dopri5",
    ):
        super(ODELSTM, self).__init__()
        d_output = d_output or d_model
        d_hidden = d_hidden or d_model
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.return_sequences = return_sequences

        self.rnn_cell = ODELSTMCell(d_model, d_hidden, solver_type=solver_type)
        self.fc = nn.Linear(self.d_hidden, self.d_output)

    def forward(self, x, state=None, timespans=None, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = [
            torch.zeros((batch_size, self.d_hidden), device=device),
            torch.zeros((batch_size, self.d_hidden), device=device),
        ]
        outputs = []
        last_output = torch.zeros((batch_size, self.d_output), device=device)

        if timespans is None:
            timespans = x.new_ones(x.shape[:-1]+(1,)) / x.shape[1]

        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            if mask is not None:
                cur_mask = mask[:, t].view(batch_size, 1)
                last_output = cur_mask * current_output + (1.0 - cur_mask) * last_output
            else:
                last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # return entire sequence
        else:
            outputs = last_output  # only last item
        return outputs, hidden_state


class IrregularSequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, t, y, mask = batch
        else:
            x, t, y = batch
            mask = None
        y_hat = self.model.forward(x, t, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = accuracy(preds, y)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            x, t, y, mask = batch
        else:
            x, t, y = batch
            mask = None
        y_hat = self.model.forward(x, t, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)

        loss = nn.CrossEntropyLoss()(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
