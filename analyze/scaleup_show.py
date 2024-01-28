import torch
import os
from matplotlib import pyplot as plot


def draw_fig(data: list, xlim=(0, 301), ylim=(68, 84), xstep=None, ystep=None, save_path="./show.jpg"):
    assert isinstance(data[0], dict)
    fig, ax = plot.subplots(dpi=300, figsize=(24, 8))
    for d in data:
        x_axis = d['x']
        y_axis = d['y']
        label = d['label']
        ax.plot(x_axis, y_axis, label=label)
        
        # for x, y  in zip(x_axis, y_axis):
        #     plot.text(x, y+0.05, y, ha='center', va= 'bottom', fontsize=7)
    
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.legend()
    if xstep is not None:
        plot.xticks(torch.arange(xlim[0], xlim[1], xstep).tolist())
    if ystep is not None:
        plot.yticks(torch.arange(ylim[0], ylim[1], ystep).tolist())
    plot.grid()
    # plot.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot.savefig(save_path)


def readlog(file=None):
    log = open(file, "r").readlines()
    log = [d.strip(" ").strip("\n") for d in log if ("img_size" in d) or ("* Acc" in d)]
    _log = []
    for i in range(len(log)):
        if "* Acc" in log[i]:
            assert "img_size" in log[i-1] 
            img_size = int(log[i-1].split("img_size")[1].strip(" ").split(" ")[0].strip(";"))
            acc1 = float(log[i].split(" ")[2])
            acc5 = float(log[i].split(" ")[4])
            _log.append(dict(size=img_size, acc1=acc1, acc5=acc5))
    _log = sorted(_log, key=lambda x: x['size'])
    x_axis = [l['size'] for l in _log]
    acc1 = [l['acc1'] for l in _log]
    acc5 = [l['acc5'] for l in _log]
    # print(x_axis, acc1, acc5)
    return _log, x_axis, acc1, acc5


def readlogflops(file=None):
    series = dict(tiny=dict(), small=dict(), base=dict())
    log = open(file, "r").readlines()
    log = [d.strip(" ").strip("\n") for d in log if ("==" in d)]
    serie: list = None
    for i in range(len(log)):
        if "= tiny =" in log[i]:
            serie = series["tiny"]
        elif "= small =" in log[i]:
            serie = series["small"]
        elif "= base =" in log[i]:
            serie = series["base"]
        
        if "= model" in log[i]:
            model = log[i].split(" ")[2]
            size = int(log[i].split(" ")[4])
            params = int(log[i].split(" ")[6]) / 1e6
            gflops = float(log[i].split(" ")[8])
            if model in serie.keys():
                serie[model].append(dict(size=size, params=params, flops=gflops))
            else:
                serie.update({model: [dict(size=size, params=params, flops=gflops)]})
    
    _log = []
    for k, v in series.items():
        models = dict()
        for _k, _v in v.items():
            model = f"{_k}_{k}"
            size = [__v["size"] for __v in _v]
            params = [__v["params"] for __v in _v]
            flops = [__v["flops"] for __v in _v]
            _log.append(dict(model=model, size=size, params=params, flops=flops))

    for x in _log:
        print(x)
    return series


scalepath = "analyze/show/scaleup.log"
readlogflops(f"{scalepath}/flops.log")
vssm_tiny = readlog(f"{scalepath}/vssmtiny_scale.log")
swin_tiny = readlog(f"{scalepath}/swintiny_scale.log")
convnext_tiny = readlog(f"{scalepath}/convnexttiny_scale.log")
deit_small = readlog(f"{scalepath}/deitsmall_scale.log")
resnet50 = readlog(f"{scalepath}/resnet50_scale.log")

vssm_small = readlog(f"{scalepath}/vssmsmall_scale.log")
swin_small = readlog(f"{scalepath}/swinsmall_scale.log")
convnext_small = readlog(f"{scalepath}/convnextsmall_scale.log")
deit_base = readlog(f"{scalepath}/deitbase_scale.log")
resnet101 = readlog(f"{scalepath}/resnet101_scale.log")

vssm_base = readlog(f"{scalepath}/vssmbase_scale.log")
swin_base = readlog(f"{scalepath}/swinbase_scale.log")
convnext_base = readlog(f"{scalepath}/convnextbase_scale.log")
replknet_31B = readlog(f"{scalepath}/replknet31b_scale.log")

print("vssm_tiny:", vssm_tiny)
print("swin_tiny:", swin_tiny)
print("convnext_tiny:", convnext_tiny)
print("deit_small:", deit_small)
print("resnet50:", resnet50)
print("=====================================")
print("vssm_small:", vssm_small)
print("swin_small:", swin_small)
print("convnext_small:", convnext_small)
print("deit_base:", deit_base)
print("resnet101:", resnet101)
print("=====================================")
print("vssm_base:", vssm_base)
print("swin_base:", swin_base)
print("convnext_base:", convnext_base)
print("replknet_31B:", replknet_31B)

if True:
    draw_fig([
        dict(x=[224, 224], y=[0, 85], label="where all the models are trained"),
        dict(x=vssm_tiny[1], y=vssm_tiny[2], label="vssm_tiny"),
        dict(x=swin_tiny[1], y=swin_tiny[2], label="swin_tiny (64, 112 is too small for windows size 7)"),
        dict(x=convnext_tiny[1], y=convnext_tiny[2], label="convnext_tiny"),
        dict(x=deit_small[1], y=deit_small[2], label="deit_small (equal to tiny model)"),
        dict(x=resnet50[1], y=resnet50[2], label="resnet50 (equal to tiny model)"),
        dict(x=vssm_tiny[1], y=vssm_tiny[2], label="vssm_tiny"),
        dict(x=replknet_31B[1], y=replknet_31B[2], label="replknet_31B (equal to *base model, 1024 is too big to test)"),
    ], xlim=(64, 1030), ylim=(0, 85), save_path="analyze/show/show_scaleup_acc.jpg")


