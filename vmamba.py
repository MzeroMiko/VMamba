##########################################################
# simplified version
# just one file and include everything  
# written by MzeroMiko                  
##########################################################

##########################################################
# usage:
# conda create -n vmamba python=3.10
# pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
# pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# python vmamba.py
##########################################################


##########################################################
# csm_triton.py
##########################################################

import torch
import warnings

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

# torch implementation ========================================
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        elif scans == 3:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = torch.rot90(x, 1, dims=(2, 3)).flatten(2, 3)
            y[:, 2, :, :] = torch.rot90(x, 2, dims=(2, 3)).flatten(2, 3)
            y[:, 3, :, :] = torch.rot90(x, 3, dims=(2, 3)).flatten(2, 3)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        elif scans == 3:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = torch.rot90(x, 1, dims=(1, 2)).flatten(1, 2)
            y[:, :, 2, :] = torch.rot90(x, 2, dims=(1, 2)).flatten(1, 2)
            y[:, :, 3, :] = torch.rot90(x, 3, dims=(1, 2)).flatten(1, 2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
        elif scans == 3:
            oy = y[:, 0, :, :].contiguous().view(B, D, -1)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3)
            y = oy
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)        
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
        elif scans == 3:
            oy = y[:, :, 0, :].contiguous().view(B, -1, D)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2)
            y = oy
            
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 3:
            y = torch.stack([
                x[:, 0, :, :, :].flatten(2, 3),
                torch.rot90(x[:, 1, :, :, :], 1, dims=(2, 3)).flatten(2, 3),
                torch.rot90(x[:, 2, :, :, :], 2, dims=(2, 3)).flatten(2, 3),
                torch.rot90(x[:, 3, :, :, :], 3, dims=(2, 3)).flatten(2, 3),
            ], dim=1)

    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)
        elif scans == 3:
            y = torch.stack([
                x[:, :, :, 0, :].flatten(1, 2),
                torch.rot90(x[:, :, :, 1, :], 1, dims=(1, 2)).flatten(1, 2),
                torch.rot90(x[:, :, :, 2, :], 2, dims=(1, 2)).flatten(1, 2),
                torch.rot90(x[:, :, :, 3, :], 3, dims=(1, 2)).flatten(1, 2),
            ], dim=1)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
        elif scans == 3:
            y = torch.stack([
                y[:, 0, :, :].contiguous().view(B, D, -1),
                torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3),
                torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3),
                torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)
        elif scans == 3:
            y = torch.stack([
                y[:, :, 0, :].contiguous().view(B, -1, D),
                torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2),
                torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2),
                torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)

        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
    
        return x, None, None, None, None


# triton implements ========================================

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor, # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h # trans
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = neg_w * DH + neg_h # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = HWRoute2 
    elif scans == 3:
        # none; rot90; rot180==flip; rot270;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = neg_w * DH + pos_h
        HWRoute2 = neg_h * DW + neg_w
        HWRoute3 = pos_w * DH + neg_h

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
    
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1  
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC        
    
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, C, H, W, NH, NW
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
        
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else CrossScanF
    if x.is_cuda:
        with torch.cuda.device(x.device):
            return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else CrossMergeF
    if y.is_cuda:
        with torch.cuda.device(y.device):
            return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
    else:
        return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)

# checks =================================================================

# class CHECK:
#     def check_csm_triton():
#         B, C, H, W = 256, 192, 56, 57
#         dtype=torch.float16
#         dtype=torch.float32
#         x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
#         y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
#         x1 = x.clone().detach().requires_grad_(True)
#         y1 = y.clone().detach().requires_grad_(True)

#         def cross_scan(x: torch.Tensor):
#             B, C, H, W = x.shape
#             L = H * W
#             xs = torch.stack([
#                 x.view(B, C, L),
#                 torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
#                 torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
#                 torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
#             ], dim=1).view(B, 4, C, L)
#             return xs
        
#         def cross_merge(out_y: torch.Tensor):
#             B, K, D, H, W = out_y.shape
#             L = H * W
#             out_y = out_y.view(B, K, D, L)
#             inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#             wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#             invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#             y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
#             return y

#         def cross_scan_1b1(x: torch.Tensor):
#             B, K, C, H, W = x.shape
#             L = H * W
#             xs = torch.stack([
#                 x[:, 0].view(B, C, L),
#                 torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
#                 torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
#                 torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
#             ], dim=1).view(B, 4, C, L)
#             return xs
        
#         def unidi_scan(x):
#             B, C, H, W = x.shape
#             x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
#             return x
        
#         def unidi_merge(ys):
#             B, K, C, H, W = ys.shape
#             return ys.view(B, 4, -1, H * W).sum(1)

#         def bidi_scan(x):
#             B, C, H, W = x.shape
#             x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
#             x = torch.cat([x, x.flip(dims=[-1])], dim=1)
#             return x
        
#         def bidi_merge(ys):
#             B, K, D, H, W = ys.shape
#             ys = ys.view(B, K, D, -1)
#             ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
#             return ys.contiguous().sum(1)

#         if True:
#             res0 = triton.testing.do_bench(lambda :cross_scan(x))
#             res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
#             # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
#             res3 = triton.testing.do_bench(lambda :cross_merge(y))
#             res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
#             # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
#             # print(res0, res1, res2, res3, res4, res5)
#             print(res0, res1, res3, res4)
#             res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
#             res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
#             # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
#             res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
#             res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
#             # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
#             # print(res0, res1, res2, res3, res4, res5)
#             print(res0, res1, res3, res4)

#         print("test cross scan")
#         for (cs0, cm0, cs1, cm1) in [
#             # channel_first -> channel_first
#             (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
#             (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
#             (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            
#             # flex: BLC->BCL; BCL->BLC; BLC->BLC;
#             (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
#             (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), out_channel_first=False)),
#             (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
            
#             # previous
#             # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
#             # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
#             # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
#         ]:
#             x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
#             o0 = cs0(x)
#             o1 = cs1(x1)
#             o0.backward(y.view(B, 4, C, H * W))
#             o1.backward(y.view(B, 4, C, H * W))
#             print((o0 - o1).abs().max())
#             print((x.grad - x1.grad).abs().max())
#             o0 = cm0(y)
#             o1 = cm1(y1)
#             o0.backward(x.view(B, C, H * W))
#             o1.backward(x.view(B, C, H * W))
#             print((o0 - o1).abs().max())
#             print((y.grad - y1.grad).abs().max())
#             x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
#             print("===============", flush=True)

#         print("test cross scan one by one")
#         for (cs0, cs1) in [
#             (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
#             # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
#         ]:
#             o0 = cs0(y)
#             o1 = cs1(y1)
#             o0.backward(y.view(B, 4, C, H * W))
#             o1.backward(y.view(B, 4, C, H * W))
#             print((o0 - o1).abs().max())
#             print((y.grad - y1.grad).abs().max())
#             x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
#             print("===============", flush=True)

#     def check_csm_scan3():
#         if False:
#             x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
#             out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
#             out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
#             out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
#             out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
#             out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
#             out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 1)
#             out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
#             out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
#             print(out1.view(4, -1))
#             print(out2.view(-1))
#             print(out3.view(4, -1))
#             print(out4.view(4, -1))
#             print(out5.view(-1, 4).t())
#             print(out6.view(-1))
#             print(out7.view(-1, 4).t())
#             print(out8.view(-1, 4).t())

#         B, C, H, W = 27, 253, 57, 58
#         x = torch.randn((B, C, H, W)).cuda()

#         for scans in [0, 1, 2, 3]:
#             o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, H, W)
#             print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
#             print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

#             kwargs = dict(in_channel_first=False, out_channel_first=False)
#             x2 = x.permute(0, 2, 3, 1).contiguous()
#             o2 = o1.permute(0, 3, 4, 1, 2).contiguous()
#             print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
#             print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

#         breakpoint()


# if __name__ == "__main__":
#     CHECK.check_csm_scan3()
#     CHECK.check_csm_triton()


##########################################################
# csms6s.py
##########################################################

import time
import torch
import warnings


WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False


def selective_scan_torch(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True, 
    *args,
    **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        # backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        # backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True,
    backend=None,
):
    fn = selective_scan_torch if backend == "torch" or (not WITH_SELECTIVESCAN_MAMBA) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


# fvcore flops =======================================
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    flops_fn = flops_selective_scan_ref if backend == "naive" else flops_selective_scan_fn
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# if __name__ == "__main__":
#     def params(B, K, C, N, L, device = torch.device("cuda"), itype = torch.float):
#         As = (-0.5 * torch.rand(K * C, N, device=device, dtype=torch.float32)).requires_grad_()
#         Bs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
#         Cs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
#         Ds = torch.randn((K * C), device=device, dtype=torch.float32).requires_grad_()
#         u = torch.randn((B, K * C, L), device=device, dtype=itype).requires_grad_()
#         delta = (0.5 * torch.rand((B, K * C, L),  device=device, dtype=itype)).requires_grad_()
#         delta_bias = (0.5 * torch.rand((K * C), device=device, dtype=torch.float32)).requires_grad_()
#         return u, delta, As, Bs, Cs, Ds, delta_bias

#     def bench(func, xs, Warmup=30, NTimes=20):
#         import time
#         torch.cuda.synchronize()
#         for r in range(Warmup):
#             for x in xs:
#                 func(x)
#         torch.cuda.synchronize()
#         tim0 = time.time()
#         for r in range(NTimes):
#             for x in xs:
#                 func(x)
#         torch.cuda.synchronize()
#         return (time.time() - tim0) / NTimes

#     def check():
#         u, delta, As, Bs, Cs, Ds, delta_bias = params(1, 4, 16, 8, 512, itype=torch.float16)
#         u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1 = [x.clone().detach().requires_grad_() for x in [u, delta, As, Bs, Cs, Ds, delta_bias]]
        
#         # out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="torch")
#         out = selective_scan_fn(u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1, True, backend="oflex")
#         out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="mamba")
#         print((out_ref - out).abs().max())
#         out.sum().backward()
#         out_ref.sum().backward()
#         for x, y in zip([u, As, Bs, Cs, Ds, delta, delta_bias], [u1, As1, Bs1, Cs1, Ds1, delta1, delta_bias1]):
#             print((x.grad - y.grad).abs().max())

#         u, delta, As, Bs, Cs, Ds, delta_bias = params(128, 4, 96, 8, 56 * 56)
#         print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="oflex"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
#         print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="mamba"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
#         print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="torch"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))

#     check()


##########################################################
# model.py
##########################################################


import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =====================================================
class Linear(nn.Linear):
    def __init__(self, *args, channel_first=False, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups
    
    def forward(self, x: torch.Tensor):
        if self.channel_first:
            # B, C, H, W = x.shape
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self_state_dict = self.state_dict()
        load_state_dict_keys = list(state_dict.keys())
        if prefix + "weight" in load_state_dict_keys:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view_as(self_state_dict["weight"])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class PatchMerge(nn.Module):
    def __init__(self, channel_first=True, in_channel_first=False, out_channel_first=False,):
        nn.Module.__init__(self)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first
        # print(f"WARNING: output [(0, 0), (1, 0), (0, 1), (1, 1)] for (H, W).")

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if not self.in_channel_first:
            B, H, W, C = x.shape
        
        if (W % 2 != 0) or (H % 2 != 0):
            PH, PW = H - H % 2, W - W % 2
            pad_shape = (PW // 2, PW - PW // 2, PH // 2, PH - PH // 2)
            pad_shape = (*pad_shape, 0, 0, 0, 0) if self.in_channel_first else (0, 0, *pad_shape, 0, 0)
            x = nn.functional.pad(x, pad_shape)
        
        xs = [
            x[..., 0::2, 0::2], x[..., 1::2, 0::2], 
            x[..., 0::2, 1::2], x[..., 1::2, 1::2],
        ] if self.in_channel_first else [
            x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :], 
            x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :],
        ]

        xs = torch.cat(xs, (1 if self.out_channel_first else -1))
        
        return xs


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# support: v01-v05; v051d,v052d,v052dc; 
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group, bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False, channel_first=True)
          
        # self.x_proj = [
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        #     for _ in range(self.k_group)
        # ]
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        # del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        # self.dt_projs.bias.data = self.dt_projs_bias.data.view(self.dt_projs.bias.shape)
        del self.dt_projs_weight
        # del self.dt_projs_bias

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if True:
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            x_dbl = self.x_proj(xs.view(B, -1, L))
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = dts.contiguous().view(B, -1, L)
            dts = self.dt_projs(dts)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class SS2D(nn.Module, SS2Dv0, SS2Dv2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self_state_dict = self.state_dict()
        self_state_dict_keys = list(self.state_dict().keys())
        load_state_dict_keys = list(state_dict.keys())
        names = {
            "x_proj_weight": "x_proj.weight", 
            "x_proj_bias": "x_proj.bias", 
            "dt_projs_weight": "dt_projs.weight", 
            "dt_projs_bias": "dt_projs.bias", 
        }
        for k, v in names.items():
            if (prefix + k in load_state_dict_keys) and (k not in self_state_dict_keys):
                assert v in self_state_dict_keys, f"{v} not in state_dict."
                state_dict[prefix + v] = state_dict[prefix + k].view_as(self_state_dict[v])
                state_dict.pop(prefix + k)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channel_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)



class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        _SS2D=SS2D,
        # =========================
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm, channel_first=self.channel_first, version=patchembed_version)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm(self.num_features, channel_first=self.channel_first), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}


    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, channel_first=False, version="v1"):
        # if channel first, then Norm and Output are both channel_first
        if version == "v1": # simple patch_embed, same with swin transformer
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
                    if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        elif version == "v2": # patch embed with stacked conv2d
            stride = patch_size // 2
            kernel_size = stride + 1
            padding = 1
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim // 2, channel_first=True) if patch_norm else nn.Identity()),
                nn.Identity(),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
                    if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        
        raise NotImplementedError

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm=True, channel_first=False, version="v1"):
        # if channel first, then Norm and Output are both channel_first
        if version == "v1": # patch merging from swin transformer
            # return PatchMerging2D(dim, 2 * dim, norm_layer, False)
            return nn.Sequential(
                PatchMerge(channel_first),
                LayerNorm(4 * dim, channel_first=channel_first) if norm else nn.Identity(),
                Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, channel_first=channel_first),
            )
        elif version == "v2": # combine pixelunshuffle and linear into conv2d
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
                    (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif version == "v3": # conv2d with overlap
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
                    (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )

        raise NotImplementedError

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        # ===========================
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
            change_name(f"layers.{i}.downsample.norm", f"layers.{i}.downsample.{1}")
            change_name(f"layers.{i}.downsample.reduction", f"layers.{i}.downsample.{2}")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["ln2d"])
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = LayerNorm(self.dims[i], channel_first=self.channel_first)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs



##########################################################
# main.py
##########################################################

from timm.models import register_model

def load_checkpoint(path="", key="model"):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    return checkpoint[key]


@register_model
def vmamba(**kwargs):
    return VSSM(**kwargs)


@register_model
def vanilla_vmamba_tiny(pretrained=False, **kwargs):
    model = VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"))
    return model


@register_model
def vanilla_vmamba_small(pretrained=False, **kwargs):
    model = VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmsmall_dp03_ckpt_epoch_238.pth"))
    return model


@register_model
def vanilla_vmamba_base(pretrained=False, **kwargs):
    model = VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmbase_dp06_ckpt_epoch_241.pth"))
    return model


@register_model
def vmamba_tiny_s2l5(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_tiny_0230_ckpt_epoch_262.pth"))
    return model


@register_model
def vmamba_small_s2l15(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"))
    return model


@register_model
def vmamba_base_s2l15(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth"))
    return model


@register_model
def vmamba_tiny_s1l8(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"))
    return model


@register_model
def vmamba_small_s1l20(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_small_0229s_ckpt_epoch_240.pth"))
    return model


@register_model
def vmamba_base_s1l20(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    if pretrained:
        model.load_state_dict(load_checkpoint("https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_base_0229s_ckpt_epoch_225.pth"))
    return model


def get_val_loader(batch_size=64, root="./val", img_size=224, sequential=True, num_workers=0):
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torchvision import transforms, datasets
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


@torch.no_grad()
def validate(data_loader, model, amp_enable=True, print_freq=100000):
    from timm.utils import accuracy, AverageMeter
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=amp_enable):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    # print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model):
    model.cuda()
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def do_validate(name="vmamba_tiny_s1l8", data="/media/memfs/ImageNet_ILSVRC2012/val"):
    from timm import create_model
    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    data_loader_val = get_val_loader(batch_size=64, root=data, num_workers=4)
    model = create_model(name, pretrained=True)
    acc1_ema, acc5_ema, loss_ema = validate(data_loader_val, model)
    print(acc1_ema, acc5_ema, loss_ema)


def do_throughput(name="vmamba_tiny_s1l8", data="/media/memfs/ImageNet_ILSVRC2012/val"):
    from timm import create_model
    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    data_loader_val = get_val_loader(batch_size=128, root=data, num_workers=4)
    model = create_model(name, pretrained=True)
    throughput(data_loader_val, model)


if __name__ == "__main__":
    # do_validate("vanilla_vmamba_tiny") # 82.17106973558698 96.03223806724185 0.7879069638634182
    # do_validate("vanilla_vmamba_small") # 83.4609923402307 96.47021178881855 0.7160880894021359
    # do_validate("vanilla_vmamba_base") # 83.72897626157689 96.62420254754197 0.6968230148378597
    # do_validate("vmamba_tiny_s2l5") # 82.48905065741832 95.99624022634936 0.7805328359985901
    # do_validate("vmamba_small_s2l15") # 83.64898106090746 96.59420434667109 0.7185911423439594
    # do_validate("vmamba_base_s2l15") # 83.87896726211686 96.71219726709586 0.7198247987933224
    # do_validate("vmamba_tiny_s1l8") # 83.87896726211686 96.71219726709586 0.7198247987933224
    # do_validate("vmamba_small_s1l20") # 83.33899965941008 96.42621442606632 nan
    # do_validate("vmamba_base_s1l20") # 83.79097254317328 96.61420314781112 0.7243299191111033
    

    # do_throughput("vanilla_vmamba_tiny")
    # do_throughput("vanilla_vmamba_small")
    # do_throughput("vanilla_vmamba_base")
    # do_throughput("vmamba_tiny_s2l5")
    # do_throughput("vmamba_small_s2l15")
    # do_throughput("vmamba_base_s2l15")
    do_throughput("vmamba_tiny_s1l8")
    # do_throughput("vmamba_small_s1l20")
    # do_throughput("vmamba_base_s1l20")
    



