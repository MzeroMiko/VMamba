optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "src.callbacks.timer.Timer",
    "params":                "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "src.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

model = {
    # Backbones from this repo
    "model":                 "src.models.sequence.backbones.model.SequenceModel",
    "unet":                  "src.models.sequence.backbones.unet.SequenceUNet",
    "sashimi":               "src.models.sequence.backbones.sashimi.Sashimi",
    "sashimi_standalone":    "models.sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm":                  "src.models.baselines.lstm.TorchLSTM",
    "gru":                   "src.models.baselines.gru.TorchGRU",
    "unicornn":              "src.models.baselines.unicornn.UnICORNN",
    "odelstm":               "src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn":          "src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn":            "src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline":   "src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn":             "src.models.baselines.samplernn.SampleRNN",
    "dcgru":                 "src.models.baselines.dcgru.DCRNNModel_classification",
    "dcgru_ss":              "src.models.baselines.dcgru.DCRNNModel_nextTimePred",
    # Baseline CNNs
    "ckconv":                "src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan":               "src.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "denseinception":        "src.models.baselines.dense_inception.DenseInception",
    "wavenet":               "src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d":        "src.models.baselines.resnet.TorchVisionResnet",  # 2D ResNet
    # Nonaka 1D CNN baselines
    "nonaka/resnet18":       "src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception":      "src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50":      "src.models.baselines.nonaka.xresnet.xresnet1d50",
    # ViT Variants (note: small variant is taken from Tri, differs from original)
    "vit":                   "models.baselines.vit.ViT",
    "vit_s_16":              "src.models.baselines.vit_all.vit_small_patch16_224",
    "vit_b_16":              "src.models.baselines.vit_all.vit_base_patch16_224",
    # Timm models
    "timm/convnext_base":    "src.models.baselines.convnext_timm.convnext_base",
    "timm/convnext_small":   "src.models.baselines.convnext_timm.convnext_small",
    "timm/convnext_tiny":    "src.models.baselines.convnext_timm.convnext_tiny",
    "timm/convnext_micro":   "src.models.baselines.convnext_timm.convnext_micro",
    "timm/resnet50":         "src.models.baselines.resnet_timm.resnet50", # Can also register many other variants in resnet_timm
    "timm/convnext_tiny_3d": "src.models.baselines.convnext_timm.convnext3d_tiny",
    # Segmentation models
    "convnext_unet_tiny":    "src.models.segmentation.convnext_unet.convnext_tiny_unet",
}

layer = {
    "id":         "src.models.sequence.base.SequenceIdentity",
    "lstm":       "src.models.baselines.lstm.TorchLSTM",
    "standalone": "models.s4.s4.S4Block",
    "s4d":        "models.s4.s4d.S4D",
    "ffn":        "src.models.sequence.modules.ffn.FFN",
    "sru":        "src.models.sequence.rnns.sru.SRURNN",
    "rnn":        "src.models.sequence.rnns.rnn.RNN",  # General RNN wrapper
    "conv1d":     "src.models.sequence.convs.conv1d.Conv1d",
    "conv2d":     "src.models.sequence.convs.conv2d.Conv2d",
    "mha":        "src.models.sequence.attention.mha.MultiheadAttention",
    "vit":        "src.models.sequence.attention.mha.VitAttention",
    "performer":  "src.models.sequence.attention.linear.Performer",
    "lssl":       "src.models.sequence.modules.lssl.LSSL",
    "s4":         "src.models.sequence.modules.s4block.S4Block",
    "fftconv":    "src.models.sequence.kernels.fftconv.FFTConv",
    "s4nd":       "src.models.sequence.modules.s4nd.S4ND",
    "mega":       "src.models.sequence.modules.mega.MegaBlock",
    "h3":         "src.models.sequence.experimental.h3.H3",
    "h4":         "src.models.sequence.experimental.h4.H4",
    # 'packedrnn': 'models.sequence.rnns.packedrnn.PackedRNN',
}

layer_decay = {
    'convnext_timm_tiny': 'src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny',
}

model_state_hook = {
    'convnext_timm_tiny_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d',
    'convnext_timm_tiny_s4nd_2d_to_3d': 'src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d',
}
