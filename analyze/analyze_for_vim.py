import os
import sys
import torch
import random
import math
from functools import partial

from utils import import_abspy, EffectiveReceiptiveField, visualize
HOME = os.environ["HOME"].rstrip("/")


class ExtraDev:
    # 5.162112298177406 30007057
    # 17.069485400571516 91157224
    def flops_s4nd(size=224, scale="ctiny"):
        import math
        from fvcore.nn import flop_count

        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./convnexts4nd")
        sys.path.insert(0, specpath)
        import timm; assert timm.__version__ == "0.5.4"
        import structured_kernels
        model1 = import_abspy("vit_all", f"{os.path.dirname(__file__)}/convnexts4nd")
        vitb = model1.vit_base_s4nd
        vitb = vitb().cuda().eval()
        model2 = import_abspy("convnext_timm", f"{os.path.dirname(__file__)}/convnexts4nd")
        ctiny = model2.convnext_tiny_s4nd
        ctiny = ctiny().cuda().eval()
        sys.path = sys.path[1:]
        # cauchy_mult makes only little difference as there's only 2-3 div operations
        
        def _supported_ops():
            def aten_fft(inputs, outputs, fftop="fft"):
                from torch.fft import fft, rfft, irfft, fftn, rfftn, irfftn 
                inp, num, dim, norm = inputs[0:4]
                is_complex = torch.is_complex(torch.tensor([], dtype=inputs[0].type().dtype()))
                shape = inp.type().sizes()

                torch._C.Value
                if isinstance(dim.type(), torch._C.IntType):
                    dim = [dim.toIValue()]
                elif isinstance(dim.type(), torch._C.TupleType):
                    dim = [d.toIValue() for d in dim.type().elements()]
                elif isinstance(dim.type(), torch._C.NoneType):
                    from torch.fft import fftn, rfftn, irfftn
                    assert fftop == "fftn"
                    if isinstance(num.type(), torch._C.NoneType):
                        dim = list(range(len(shape)))
                    elif isinstance(num.type(), torch._C.ListType):
                        assert isinstance(num.type().getElementType(), torch._C.IntType)
                        num_dims = len(tuple(num.node().inputs()))
                        guess_dim = [-1 - i for i in range(num_dims)]
                        print(f"Warning, We are not sure about this, guess the fft dim are {guess_dim}. input: {shape}, n or s: {num}, dim: {dim.type()}")
                        dim = guess_dim
                else:
                    raise NotImplementedError
                
                flops = math.prod(shape) * math.prod([math.log2(shape[i]) for i in dim]) * (4 if is_complex else 1)
                # print(flops, dim, is_complex, shape)
                return flops
            supported_ops={
                "aten::fft_fft": aten_fft,
                "aten::fft_rfft": aten_fft,
                "aten::fft_rfftn": partial(aten_fft, fftop="fftn"),
                "aten::fft_irfft": aten_fft,
                "aten::fft_irfftn": partial(aten_fft, fftop="fftn"),
            }
            return supported_ops

        model = {"ctiny": ctiny, "vitb": vitb}[scale]
        input_shape = (1, 3, size, size)
        inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)
        model(inputs[0]) # to force init first
        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=_supported_ops())
        print("GFlops: ", sum(Gflops.values()), "Params: ", sum([p.numel() for _, p in model.named_parameters()]), flush=True)

    def build_vim_for_throughput(with_ckpt=False, remove_head=False, only_backbone=False, size=224):
        img_size = size
        imgHW = int(math.sqrt(img_size))
        specpath = f"{HOME}/packs/Vim/mamba-1p1p1"
        sys.path.insert(0, specpath)
        import mamba_ssm
        _model = import_abspy("models_mamba", f"{HOME}/packs/Vim/vim")
        sys.path = sys.path[1:]
        # model = _model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        kwargs=dict()
        # model = _model.VisionMamba(patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
        
        model = _model.VisionMamba(img_size=img_size, patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
           
        if only_backbone:
            # copy from https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py#VisionMamba
            # added "return hidden_states, token_position"
            RMSNorm, layer_norm_fn, rms_norm_fn = _model.RMSNorm, _model.layer_norm_fn, _model.rms_norm_fn
            def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
                # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
                # with slight modifications to add the dist_token
                x = self.patch_embed(x)
                B, M, _ = x.shape

                if self.if_cls_token:
                    if self.use_double_cls_token:
                        cls_token_head = self.cls_token_head.expand(B, -1, -1)
                        cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                        token_position = [0, M + 1]
                        x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                        M = x.shape[1]
                    else:
                        if self.use_middle_cls_token:
                            cls_token = self.cls_token.expand(B, -1, -1)
                            token_position = M // 2
                            # add cls token in the middle
                            x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                        elif if_random_cls_token_position:
                            cls_token = self.cls_token.expand(B, -1, -1)
                            token_position = random.randint(0, M)
                            x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                            print("token_position: ", token_position)
                        else:
                            cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                            token_position = 0
                            x = torch.cat((cls_token, x), dim=1)
                        M = x.shape[1]

                if self.if_abs_pos_embed:
                    # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
                    #     x = x + self.pos_embed
                    # else:
                    #     pos_embed = interpolate_pos_embed_online(
                    #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
                    #             )
                    x = x + self.pos_embed
                    x = self.pos_drop(x)

                if if_random_token_rank:

                    # 生成随机 shuffle 索引
                    shuffle_indices = torch.randperm(M)

                    if isinstance(token_position, list):
                        print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
                    else:
                        print("original value: ", x[0, token_position, 0])
                    print("original token_position: ", token_position)

                    # 执行 shuffle
                    x = x[:, shuffle_indices, :]

                    if isinstance(token_position, list):
                        # 找到 cls token 在 shuffle 之后的新位置
                        new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                        token_position = new_token_position
                    else:
                        # 找到 cls token 在 shuffle 之后的新位置
                        token_position = torch.where(shuffle_indices == token_position)[0].item()

                    if isinstance(token_position, list):
                        print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
                    else:
                        print("new value: ", x[0, token_position, 0])
                    print("new token_position: ", token_position)




                if_flip_img_sequences = False
                if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
                    x = x.flip([1])
                    if_flip_img_sequences = True

                # mamba impl
                residual = None
                hidden_states = x
                if not self.if_bidirectional:
                    for layer in self.layers:

                        if if_flip_img_sequences and self.if_rope:
                            hidden_states = hidden_states.flip([1])
                            if residual is not None:
                                residual = residual.flip([1])

                        # rope about
                        if self.if_rope:
                            hidden_states = self.rope(hidden_states)
                            if residual is not None and self.if_rope_residual:
                                residual = self.rope(residual)

                        if if_flip_img_sequences and self.if_rope:
                            hidden_states = hidden_states.flip([1])
                            if residual is not None:
                                residual = residual.flip([1])

                        hidden_states, residual = layer(
                            hidden_states, residual, inference_params=inference_params
                        )
                else:
                    # get two layers in a single for-loop
                    for i in range(len(self.layers) // 2):
                        if self.if_rope:
                            hidden_states = self.rope(hidden_states)
                            if residual is not None and self.if_rope_residual:
                                residual = self.rope(residual)

                        hidden_states_f, residual_f = self.layers[i * 2](
                            hidden_states, residual, inference_params=inference_params
                        )
                        hidden_states_b, residual_b = self.layers[i * 2 + 1](
                            hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                        )
                        hidden_states = hidden_states_f + hidden_states_b.flip([1])
                        residual = residual_f + residual_b.flip([1])

                if not self.fused_add_norm:
                    if residual is None:
                        residual = hidden_states
                    else:
                        residual = residual + self.drop_path(hidden_states)
                    hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states = fused_add_norm_fn(
                        self.drop_path(hidden_states),
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )

                return hidden_states, token_position
            
                # return only cls token if it exists
                if self.if_cls_token:
                    if self.use_double_cls_token:
                        return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
                    else:
                        if self.use_middle_cls_token:
                            return hidden_states[:, token_position, :]
                        elif if_random_cls_token_position:
                            return hidden_states[:, token_position, :]
                        else:
                            return hidden_states[:, token_position, :]

                if self.final_pool_type == 'none':
                    return hidden_states[:, -1, :]
                elif self.final_pool_type == 'mean':
                    return hidden_states.mean(dim=1)
                elif self.final_pool_type == 'max':
                    return hidden_states
                elif self.final_pool_type == 'all':
                    return hidden_states
                else:
                    raise NotImplementedError

            # modified from https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py#VisionMamba
            def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
                hs, token_position = forward_features(self, x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
                print("self.if_cls_token", self.if_cls_token, end=" ")
                print("self.use_double_cls_token", self.use_double_cls_token, end=" ")
                print("self.use_middle_cls_token", self.use_middle_cls_token, end=" ")
                print("if_random_cls_token_position", if_random_cls_token_position, end=" ")
                print("if_random_token_rank", if_random_token_rank, end=" ")

                indexes = list(range(hs.shape[1]))
                token_position = token_position if isinstance(token_position, list) else [token_position]
                for t in token_position:
                    indexes.remove(t)
                hs = hs[:, indexes, :].contiguous()
                H = int(math.sqrt(hs.shape[1]))
                hs = hs.permute(0, 2, 1).contiguous().view(hs.shape[0], -1, H, H)
                return hs

            model.forward = partial(forward, model)

        elif remove_head:
            model.forward = partial(model.forward, return_features=True)


        model = model.cuda().eval()
        
        if with_ckpt:
            ckpt = torch.load(open(f"{HOME}/packs/ckpts/vim_s_midclstok_80p5acc.pth", "rb"), map_location=torch.device("cpu"))["model"]
        
            # to interplate pos_mebed, the cls_token position must be fixed !
            # otherwise, ignore cls_token and apply interplation to all
            # this checkpoint uses middle cls token
            # from mmpretrain.models.backbones.vision_transformer import resize_pos_embed
            assert not model.use_double_cls_token
            assert model.use_middle_cls_token
            assert ckpt["pos_embed"].shape[1] == 197
            target_token_length = (img_size // 16)**2
            target_token_length_HW = ((img_size // 16), (img_size // 16))
            if target_token_length != 197 - 1:
                mid_token_idx = target_token_length // 2
                cls_token = ckpt["pos_embed"][:, 83:84, :]
                extra_tokens_left = ckpt["pos_embed"][:, :83, :]
                extra_tokens_right = ckpt["pos_embed"][:, 84:, :]
                extra_tokens = torch.cat([extra_tokens_left, extra_tokens_right], dim=1)
                extra_tokens = extra_tokens.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
                extra_tokens = torch.nn.functional.interpolate(extra_tokens, size=target_token_length_HW, align_corners=False, mode="bicubic")
                extra_tokens = extra_tokens.permute(0, 2, 3, 1).contiguous().view(1, target_token_length, -1)
                pos_embed = torch.cat([extra_tokens[:, :mid_token_idx, :], cls_token, extra_tokens[:, mid_token_idx:, :]], dim=1)
                ckpt["pos_embed"] = pos_embed
            
            model.load_state_dict(ckpt)

        return model

    # 5.301500928 25796584
    def flops_vim(size=224):
        from fvcore.nn import flop_count

        # FLOPs.fvcore_flop_count(BuildModels.build_vmamba(scale="tv2").cuda().eval(), input_shape=(3, size, size), show_arch=False)
        specpath = f"{HOME}/packs/Vim/mamba-1p1p1"
        sys.path.insert(0, specpath)
        import mamba_ssm
        _model = import_abspy("models_mamba", f"{HOME}/packs/Vim/vim")
        sys.path = sys.path[1:]
        # model = _model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        kwargs=dict()
        # model = _model.VisionMamba(patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
        
        # fused add norm share the same flops as naive one
        model = _model.VisionMamba(img_size=size, patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=False, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
        vims = model.cuda().eval()

        # RMSNorm share the same flops as naive one
        # https://github.com/state-spaces/mamba/blob/v1.2.2/mamba_ssm/ops/triton/layernorm.py
        def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
            dtype = x.dtype
            if upcast:
                weight = weight.float()
                bias = bias.float() if bias is not None else None
            if upcast:
                x = x.float()
                residual = residual.float() if residual is not None else residual
            if residual is not None:
                x = (x + residual).to(x.dtype)
            rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
            out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
            out = out.to(dtype)
            return out if not prenorm else (out, x)

        def rms_forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
            return rms_norm_ref(
                x,
                self.weight,
                self.bias,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                upcast=residual_in_fp32,
            )

        for k, m in vims.named_modules():
            if isinstance(m, _model.RMSNorm):
                m.forward = partial(rms_forward, m)

        input_shape = (1, 3, size, size)
        model = vims.cuda().eval()
        import math

        def causal_conv_1d_jit(inputs, outputs):
            """
            https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
            x: (batch, dim, seqlen) weight: (dim, width) bias: (dim,) out: (batch, dim, seqlen)
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            """
            from fvcore.nn.jit_handles import conv_flop_jit
            return conv_flop_jit(inputs, outputs)

        # ONLY FOR VisionMamba
        def MambaInnerFnNoOutProj_jit(inputs, outputs):
            """
            conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
            x, z = xz.chunk(2, dim=1)
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, True)
            x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
            B = x_dbl[:, delta_rank:delta_rank + d_state]
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            C = x_dbl[:, -d_state:]
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            out, scan_intermediates, out_z = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
            )
            """
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, A = inputs[0:6]
            Batch, _, L = xz.type().sizes()
            CWidth = conv1d_weight.type().sizes()[-1]
            H = A.type().sizes()[-1] # 16
            Dim, R = delta_proj_weight.type().sizes()
            assert tuple(xz.type().sizes()) == (Batch, 2 * Dim, L)
            assert tuple(conv1d_weight.type().sizes()) == (Dim, 1, CWidth)
            assert tuple(x_proj_weight.type().sizes()) == (R + H + H, Dim)
            assert tuple(A.type().sizes()) == (Dim, H)

            with_Z = True
            with_D = False
            if "D" in inputs[6].debugName():
                assert tuple(inputs[6].type().sizes()) == (Dim,)
                with_D = True

            flops = 0
            flops += Batch * (Dim * L) * CWidth # causal_conv1d_cuda.causal_conv1d_fwd
            flops += Batch * (Dim * L) * (R + H + H) # x_dbl = F.linear(...
            flops += Batch * (Dim * R) * (L) # delta_proj_weight @ x_dbl[:, :delta_rank]
            
            # https://github.com/state-spaces/mamba/issues/110
            flops = 9 * Batch * L * Dim * H
            if with_D:
                flops += Batch * Dim * L
            if with_Z:
                flops += Batch * Dim * L  

            return flops

        # ONLY FOR Mamba
        def MambaInnerFn_jit(inputs, outputs):
            """
            conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
            x, z = xz.chunk(2, dim=1)
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, True)
            x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
            B = x_dbl[:, delta_rank:delta_rank + d_state]
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            C = x_dbl[:, -d_state:]
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            out, scan_intermediates, out_z = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
            )
            F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            """
            xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, out_proj_weight, out_proj_bias, A = inputs[0:8]
            Batch, _, L = xz.type().sizes()
            CWidth = conv1d_weight.type().sizes()[-1]
            H = A.type().sizes()[-1] # 16
            Dim, R = delta_proj_weight.type().sizes()
            assert tuple(xz.type().sizes()) == (Batch, 2 * Dim, L)
            assert tuple(conv1d_weight.type().sizes()) == (Dim, 1, CWidth)
            assert tuple(x_proj_weight.type().sizes()) == (R + H + H, Dim)
            assert tuple(A.type().sizes()) == (Dim, H)

            with_Z = True
            with_D = False
            if "D" in inputs[6].debugName():
                assert tuple(inputs[6].type().sizes()) == (Dim,)
                with_D = True

            flops = 0
            flops += Batch * (Dim * L) * CWidth # causal_conv1d_cuda.causal_conv1d_fwd
            flops += Batch * (Dim * L) * (R + H + H) # x_dbl = F.linear(...
            flops += Batch * (Dim * R) * (L) # delta_proj_weight @ x_dbl[:, :delta_rank]
            
            # https://github.com/state-spaces/mamba/issues/110
            flops = 9 * Batch * L * Dim * H
            if with_D:
                flops += Batch * Dim * L
            if with_Z:
                flops += Batch * Dim * L  

            out_weight_shape = out_proj_weight.type().sizes()
            assert out_proj_weight[1] == Dim
            flops += Batch * Dim * L * out_proj_weight[0]

            return flops

        supported_ops={
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.CausalConv1dFn": causal_conv_1d_jit,
            "prim::PythonOp.MambaInnerFnNoOutProj": MambaInnerFnNoOutProj_jit,
            "prim::PythonOp.MambaInnerFn": MambaInnerFn_jit,
        }
        inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)
        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
        # print(Gflops.items())
        print("GFlops: ", sum(Gflops.values()), "Params: ", sum([p.numel() for _, p in model.named_parameters()]), flush=True)

    def erf_vim(data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"):
        print("vim ================================", flush=True)
        specpath = f"{HOME}/packs/Vim/mamba-1p1p1"
        sys.path.insert(0, specpath)
        import mamba_ssm
        _model = import_abspy("models_mamba", f"{HOME}/packs/Vim/vim")
        sys.path = sys.path[1:]
        # model = _model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        kwargs=dict()
        # model = _model.VisionMamba(patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
        
        model = _model.VisionMamba(img_size=1024, patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
           
        # copy from https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py#VisionMamba
        # added "return hidden_states, token_position"
        RMSNorm, layer_norm_fn, rms_norm_fn = _model.RMSNorm, _model.layer_norm_fn, _model.rms_norm_fn
        def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
            # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
            # with slight modifications to add the dist_token
            x = self.patch_embed(x)
            B, M, _ = x.shape

            if self.if_cls_token:
                if self.use_double_cls_token:
                    cls_token_head = self.cls_token_head.expand(B, -1, -1)
                    cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                    token_position = [0, M + 1]
                    x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                    M = x.shape[1]
                else:
                    if self.use_middle_cls_token:
                        cls_token = self.cls_token.expand(B, -1, -1)
                        token_position = M // 2
                        # add cls token in the middle
                        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    elif if_random_cls_token_position:
                        cls_token = self.cls_token.expand(B, -1, -1)
                        token_position = random.randint(0, M)
                        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                        print("token_position: ", token_position)
                    else:
                        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                        token_position = 0
                        x = torch.cat((cls_token, x), dim=1)
                    M = x.shape[1]

            if self.if_abs_pos_embed:
                # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
                #     x = x + self.pos_embed
                # else:
                #     pos_embed = interpolate_pos_embed_online(
                #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
                #             )
                x = x + self.pos_embed
                x = self.pos_drop(x)

            if if_random_token_rank:

                # 生成随机 shuffle 索引
                shuffle_indices = torch.randperm(M)

                if isinstance(token_position, list):
                    print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
                else:
                    print("original value: ", x[0, token_position, 0])
                print("original token_position: ", token_position)

                # 执行 shuffle
                x = x[:, shuffle_indices, :]

                if isinstance(token_position, list):
                    # 找到 cls token 在 shuffle 之后的新位置
                    new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                    token_position = new_token_position
                else:
                    # 找到 cls token 在 shuffle 之后的新位置
                    token_position = torch.where(shuffle_indices == token_position)[0].item()

                if isinstance(token_position, list):
                    print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
                else:
                    print("new value: ", x[0, token_position, 0])
                print("new token_position: ", token_position)




            if_flip_img_sequences = False
            if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
                x = x.flip([1])
                if_flip_img_sequences = True

            # mamba impl
            residual = None
            hidden_states = x
            if not self.if_bidirectional:
                for layer in self.layers:

                    if if_flip_img_sequences and self.if_rope:
                        hidden_states = hidden_states.flip([1])
                        if residual is not None:
                            residual = residual.flip([1])

                    # rope about
                    if self.if_rope:
                        hidden_states = self.rope(hidden_states)
                        if residual is not None and self.if_rope_residual:
                            residual = self.rope(residual)

                    if if_flip_img_sequences and self.if_rope:
                        hidden_states = hidden_states.flip([1])
                        if residual is not None:
                            residual = residual.flip([1])

                    hidden_states, residual = layer(
                        hidden_states, residual, inference_params=inference_params
                    )
            else:
                # get two layers in a single for-loop
                for i in range(len(self.layers) // 2):
                    if self.if_rope:
                        hidden_states = self.rope(hidden_states)
                        if residual is not None and self.if_rope_residual:
                            residual = self.rope(residual)

                    hidden_states_f, residual_f = self.layers[i * 2](
                        hidden_states, residual, inference_params=inference_params
                    )
                    hidden_states_b, residual_b = self.layers[i * 2 + 1](
                        hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                    )
                    hidden_states = hidden_states_f + hidden_states_b.flip([1])
                    residual = residual_f + residual_b.flip([1])

            if not self.fused_add_norm:
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual + self.drop_path(hidden_states)
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            return hidden_states, token_position
        
            # return only cls token if it exists
            if self.if_cls_token:
                if self.use_double_cls_token:
                    return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
                else:
                    if self.use_middle_cls_token:
                        return hidden_states[:, token_position, :]
                    elif if_random_cls_token_position:
                        return hidden_states[:, token_position, :]
                    else:
                        return hidden_states[:, token_position, :]

            if self.final_pool_type == 'none':
                return hidden_states[:, -1, :]
            elif self.final_pool_type == 'mean':
                return hidden_states.mean(dim=1)
            elif self.final_pool_type == 'max':
                return hidden_states
            elif self.final_pool_type == 'all':
                return hidden_states
            else:
                raise NotImplementedError

        # modified from https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py#VisionMamba
        def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
            hs, token_position = forward_features(self, x, inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
            print("self.if_cls_token", self.if_cls_token, end=" ")
            print("self.use_double_cls_token", self.use_double_cls_token, end=" ")
            print("self.use_middle_cls_token", self.use_middle_cls_token, end=" ")
            print("if_random_cls_token_position", if_random_cls_token_position, end=" ")
            print("if_random_token_rank", if_random_token_rank, end=" ")

            indexes = list(range(hs.shape[1]))
            token_position = token_position if isinstance(token_position, list) else [token_position]
            for t in token_position:
                indexes.remove(t)
            hs = hs[:, indexes, :].contiguous()
            H = int(math.sqrt(hs.shape[1]))
            hs = hs.permute(0, 2, 1).contiguous().view(hs.shape[0], -1, H, H)
            return hs

        model.forward = partial(forward, model)
        
        vims = model.cuda().eval()
        model_before = EffectiveReceiptiveField.get_input_grad_avg(vims, size=1024, data_path=data_path, norms=EffectiveReceiptiveField.simpnorm)

        # with ckpt
        ckpt = torch.load(open(f"{HOME}/packs/ckpts/vim_s_midclstok_80p5acc.pth", "rb"), map_location=torch.device("cpu"))["model"]
        
        # to interplate pos_mebed, the cls_token position must be fixed !
        # otherwise, ignore cls_token and apply interplation to all
        # this checkpoint uses middle cls token
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
        assert not vims.use_double_cls_token
        assert vims.use_middle_cls_token
        assert ckpt["pos_embed"].shape[1] == 197
        cls_token = ckpt["pos_embed"][:, 83:84, :]
        extra_tokens_left = ckpt["pos_embed"][:, :83, :]
        extra_tokens_right = ckpt["pos_embed"][:, 84:, :]
        extra_tokens = torch.cat([extra_tokens_left, extra_tokens_right], dim=1)
        extra_tokens = extra_tokens.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        extra_tokens = torch.nn.functional.interpolate(extra_tokens, size=(64, 64), align_corners=False, mode="bicubic")
        extra_tokens = extra_tokens.permute(0, 2, 3, 1).contiguous().view(1, 4096, -1)
        pos_embed = torch.cat([extra_tokens[:, :2048, :], cls_token, extra_tokens[:, 2048:, :]], dim=1)
        ckpt["pos_embed"] = pos_embed
        
        model.load_state_dict(ckpt)
        vims = model.cuda().eval()
        model_after = EffectiveReceiptiveField.get_input_grad_avg(vims, size=1024, data_path=data_path, norms=EffectiveReceiptiveField.simpnorm)
        return model_before, model_after


if __name__ == "__main__":
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show").rstrip("/")
    data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"
    
    ExtraDev.flops_vim()
    ExtraDev.flops_s4nd()

    vim_before, vim_after = ExtraDev.erf_vim()
    visualize.visualize_snsmaps([(vim_before, ""), (vim_after, "")], savefig=f"{showpath}/erf_s4ndmethods.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

