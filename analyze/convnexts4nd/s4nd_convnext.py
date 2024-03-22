# s4/configs/experiment/s4nd/convnext/convnext_timm_tiny_s4nd_imagenet.yaml
"""
model:
  img_size: ${dataset.__l_max}
  drop_path_rate: 0.1
  patch_size: 4  # 2 or 4, use for stem downsample factor
  stem_channels: 32  # only used for s4nd stem currently
  stem_type: new_s4nd_patch  # options: patch (regular convnext), s4nd_patch, new_s4nd_patch (best), s4nd
  stem_l_max: [16, 16]  # stem_l_max=None,  # len of l_max in stem (if using s4)
  downsample_type: s4nd  # eg, s4nd, null (for regular strided conv)
  downsample_act: false
  downsample_glu: True
  conv_mlp: false
  custom_ln: false # only used if conv_mlp=1, should benchmark to make sure this is faster/more mem efficient, also need to turn off weight decay
  layer:  # null means use regular conv2d in convnext
    _name_: s4nd
    d_state: 64
    channels: 1
    bidirectional: true
    activation: null  # mimics convnext style
    final_act: none
    initializer: null
    weight_norm: false
    dropout: 0
    tie_dropout: ${oc.select:model.tie_dropout,null}
    init: fourier
    rank: 1
    trank: 1
    dt_min: 0.01
    dt_max: 1.0
    lr: 0.001
    # length_correction: true
    n_ssm: 1
    deterministic: false # Special C init
    l_max: ${oc.select:dataset.__l_max,null} # Grab dataset length if exists, otherwise set to null and kernel will automatically resize
    verbose: true
    linear: true
    return_state: false
    bandlimit: null
    contract_version: 0  # 0 is for 2d, 1 for 1d or 3d (or other)
  stem_layer:
    dt_min: 0.1
    dt_max: 1.0
    init: fourier
  stage_layers:
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
"""

# s4/configs/model/layer/s4nd.yaml
"""
_name_: s4nd
d_state: 64
channels: 1
bidirectional: true
activation: gelu
final_act: glu
initializer: null
weight_norm: false
trank: 1
dropout: ${..dropout} # Same as null
tie_dropout: ${oc.select:model.tie_dropout,null}
init: legs
rank: 1
dt_min: 0.001
dt_max: 0.1
lr:
  dt: 0.001
  A: 0.001
  B: 0.001
n_ssm: 1
deterministic: false # Special C init
l_max: ${oc.select:dataset.__l_max,null} # Grab dataset length if exists, otherwise set to null and kernel will automatically resize
verbose: true
linear: false
"""



