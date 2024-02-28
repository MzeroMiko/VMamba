# mamba-mini
An efficient implementation of selective scan in one file, works with both cpu and gpu, with corresponding mathematical derivation. It is probably the code which is the most close to selective_scan_cuda in mamba.

### mathematical derivation
![image](../assets/derivation.png)

### code
```python
import torch
def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
    """
    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us, dts: (L, B, G, D) # L is chunk_size
        As: (G, D, N)
        Bs, Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (B, G, D, N)
        """
        ts = dts.cumsum(dim=0)
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
        scale = Ats[-1].detach()
        rAts = Ats / scale
        duts = dts * us
        dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)
        hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0) 
        hs = hs_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) 
        return ys, hs
    
    inp_dtype = us.dtype
    has_D = Ds is not None

    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if has_D else None
    D = As.shape[1]
    
    oys = []
    # ohs = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
    for i in range(0, L - 1, chunksize):
        ys, hs = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize], 
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix, 
        )
        oys.append(ys)
        # ohs.append(hs)
        hprefix = hs[-1]

    oys = torch.cat(oys, dim=0)
    # ohs = torch.cat(ohs, dim=0)
    if has_D:
        oys = oys + Ds * us
    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    oys = oys.to(inp_dtype)
    # hprefix = hprefix.to(inp_dtype)

    return oys if not return_last_state else (oys, hprefix.view(B, G * D, N))

```

### to test
```bash
pytest test_selective_scan.py
```
