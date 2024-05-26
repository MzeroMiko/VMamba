## :white_check_mark: Updates
* **`May. 26th, 2024`**: Update: we release the updated weights of VMambav2, together with the new arxiv paper.

* **`May. 7th, 2024`**: Update: **Important!** using `torch.backends.cudnn.enabled=True` in downstream tasks may be quite slow. If you found vmamba quite slow in your machine, disable it in vmamba.py, else, ignore this.

* **` April. 10th, 2024`**: Update: we have released [arXiv 2401.10166v2](https://arxiv.org/abs/2401.10166v2), which contains lots of updates we made related to VMambav2!
 
* **` March. 20th, 2024`**: Update: we have released the `configs/logs/checkpoints` for `classification/detection/segmentation` of VMambav2. We'are still working on VMambav3! 

* **` March. 16th, 2024`**: Improvement: we implemented models with channel_first data layout, which GREATLY raises the `throughput` of the model on A100 (On V100, due to the slow implementation of F.conv2d compared to F.linear, it would not speed up.), Try using `norm_layer="ln2d"` (when inferencing or training) rather than `norm_layer="ln"` to unlock this feature with almost no performance cost!

* **` March. 8th, 2024`**: Update + Improvement: we update the performance of `VMamba-T`, `Vmamba-S`, `VMamba-B` with nightly build, checkpoints and logs are coming soon. (Note that these models are trained without `CrossScanTriton` or `forwardtype=v4`, you can modify those configs yourself to raise the speed with almost no cost!)

* **` March. 8th, 2024`**: Improvement: we implemented `CrossScan` and `CrossMerge` in `triton`, which speed the training up again. `CrossScan` and `CrossMerge` implemented in triton is ~2x faster than implemented in pytorch. Meanwhile, use `v4` rather than `v3` or `v2` in forwardtype also raise the speed GREATLY!.

* **` Feb. 26th, 2024`:** Improvement: we now support flexible output of `selective scan`. That means whatever type the input is, the output can always be float32. The feature is useful as when training with float16, the loss often get nan due to the overflow over float16. In the meantime, training with float32 costs more time. Input with float16 and output with float32 can be fast, but in the meantime, the loss is less likely to be NaN.   Try `SelectiveScanOflex` with float16 input and float32 output to enjoy that feature!

* **` Feb. 22th, 2024`:** Pre-Release: we set a pre-release to share nightly-build checkpoints in classificaion. Feel free to enjoy those new features with faster code and higher performance! 

* **` Feb. 18th, 2024`:** Release: all the checkpoints and logs of `VMamba` (`VSSM version 0`) in classification have been released. These checkpoints correspond to the experiments done before date #20240119, if there is any mismatch to the latest code in main, please let me know, and I'll fix that. This is related to issue#1 and issue#37.

* **` Feb. 16th, 2024`:** Fix bug + Improvement: `SS2D.forward_corev1` is deprecated. Fixed some bugs related to issue#30 (in test_selective scan.py, we now compare `ours` with `mamba_ssm` rather than `selective_scan_ref`), issue#32, issue#31. `backward nrow` has been added and tested in selective_scan.

* **` Feb. 4th, 2024`:** Fix bug + Improvement: Do not use `SS2D.forward_corev1` with `float32=False` for training (testing is ok), as it's unstable training in float16 for selective scan. We released `SS2D.forward_corev2`, which is in float32, and is faster than `SS2D.forward_corev1`.

* **` Feb. 1st, 2024`:** Fix bug: we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).

* **` Jan. 31st, 2024`:** ~~Add feature: `selective_scan` now supports an extra argument `nrow` in `[1, 2, 4]`. If you find your device is strong and the time consumption keeps as `d_state` rises, try this feature to speed up `nrows` x without any cost ! Note this feature is actually a `bug fix` for [mamba](https://github.com/state-spaces/mamba).~~

* **` Jan. 28th, 2024`:** Add feature: we cloned main into a new branch called `20240128-achieve`, the main branch has experienced a great update now. The code now are much easier to use in your own project, and the training speed is faster! This new version is totally compatible with original one, and you can use previous checkpoints without any modification. But if you want to use exactly the same models as original ones, just change `forward_core = self.forward_corev1` into `forward_core = self.forward_corev0` in `classification/models/vmamba/vmamba.py#SS2D` or you can change into the branch `20240128-archive` instead.

* **` Jan. 23th, 2024`:** Add feature:  we add an alternative for mamba_ssm and causal_conv1d. Typing `pip install .` in `selective_scan` and you can get rid of those two packages. ~~Just turn `self.forward_core = self.forward_corev0` to `self.forward_core = self.forward_corev1` in `classification/models/vmamba/vmamba.py#SS2D.__init__` to enjoy that feature.~~ The training speed is expected to raise from 20min/epoch for tiny in 8x4090GPU to 17min/epoch, GPU memory cost reduces too.

* **` Jan. 22th, 2024`:** We have released VMamba-T/S pre-trained weights. The ema weights should be converted before transferring to downstream tasks to match the module names using [get_ckpt.py](analyze/get_ckpt.py).

* **` Jan. 19th, 2024`:** The source code for classification, object detection, and semantic segmentation are provided. 