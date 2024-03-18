# Analyze log ==================================
import os
import torch

def get_acc_convnext(f: list):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    emaaccs = []
    accs = []
    for i, line in enumerate(f):
        if "* Acc" in line and ("Accuracy of the model EMA on" in f[i + 1]):
            l: str = line.strip(" ").split(" ") # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            emaaccs.append(dict(acc1=float(l[2]), acc5=float(l[4]))) 
        elif "* Acc" in line and ("Accuracy of the model on" in f[i + 1]):
            l: str = line.strip(" ").split(" ") # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            accs.append(dict(acc1=float(l[2]), acc5=float(l[4]))) 
    
    accs = dict(acc1=[a['acc1'] for a in accs], acc5=[a['acc5'] for a in accs])
    emaaccs = dict(acc1=[a['acc1'] for a in emaaccs], acc5=[a['acc5'] for a in emaaccs])
    x_axis = range(len(accs['acc1']))  
    return x_axis, accs, emaaccs


def get_loss_convnext(f: list, x1e=torch.tensor(list(range(0, 625, 10)) + [624]).view(1, -1) / 625, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    avglosses = []
    losses = []
    for i, line in enumerate(f):
        if "Epoch: [" in line and ("loss:" in line):
            l = line.split("loss:")[1].strip(" ").split(" ")[:2]
            losses.append(float(l[0]))
            avglosses.append(float(l[1].split(")")[0].strip("()")))

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]
    avglosses = [l * scale for l in avglosses]

    return x_axis, losses, avglosses


def get_acc_swin(f: list, split_ema=False):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    emaaccs = None
    accs = []
    for i, line in enumerate(f):
        if "* Acc" in line:
            l: str = line.split("INFO")[-1].strip(" ").split(" ") # [*, Acc@1, 0.642, Acc@5, 2.780, ...]
            accs.append(dict(acc1=float(l[2]), acc5=float(l[4]))) 
    accs = dict(acc1=[a['acc1'] for a in accs], acc5=[a['acc5'] for a in accs])
    if split_ema:
        emaaccs = dict(acc1=[a for i, a in enumerate(accs['acc1']) if i % 2 == 1], 
                       acc5=[a for i, a in enumerate(accs['acc5']) if i % 2 == 1])
        accs = dict(acc1=[a for i, a in enumerate(accs['acc1']) if i % 2 == 0], 
                       acc5=[a for i, a in enumerate(accs['acc5']) if i % 2 == 0])
    x_axis = range(len(accs['acc1']))  
    return x_axis, accs, emaaccs


def get_loss_swin(f: list, x1e=torch.tensor(list(range(0, 1253, 10))).view(1, -1) / 1253, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    avglosses = []
    losses = []
    for i, line in enumerate(f):
        if "Train: [" in line and ("loss" in line):
            l = line.split("loss")[1].strip(" ").split(" ")[:2]
            losses.append(float(l[0]))
            avglosses.append(float(l[1].split(")")[0].strip("()")))

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]
    avglosses = [l * scale for l in avglosses]

    return x_axis, losses, avglosses


def get_acc_mmpretrain(f: list):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    accs = []
    for i, line in enumerate(f):
        if "accuracy_top-1" in line:
            line = line.split("accuracy_top-1")[1] # ": 81.182, "accuracy_top-5": 95.606}
            lis = line.split("accuracy_top-5") # [": 81.182, ", ": 95.606}]
            acc1 = float(lis[0].split(",")[0].split(" ")[-1])
            acc5 = float(lis[1].split("}")[0].split(" ")[-1])
            accs.append(dict(acc1=acc1, acc5=acc5))
    accs = dict(acc1=[a['acc1'] for a in accs], acc5=[a['acc5'] for a in accs])
    x_axis = list(range(10, 10 * len(accs['acc1']) + 1, 10))  
    return x_axis, accs, None


def get_loss_mmpretrain(f: list, x1e=torch.tensor(list(range(100, 1201, 100))).view(1, -1) / 1201, scale=1):
    if isinstance(f, str):
        f = open(f, "r").readlines()

    losses = []
    for i, line in enumerate(f):
        if "loss" in line:
            line = line.split("loss")[1].split(",")[0].split(" ")[-1] # 6.95273
            losses.append(float(line))

    x = x1e
    x = x.repeat(len(losses) // x.shape[1] + 1, 1)
    x = x + torch.arange(0, x.shape[0]).view(-1, 1)
    x = x.flatten().tolist()
    x_axis = x[:len(losses)]

    losses = [l * scale for l in losses]
    # avglosses = [l * scale for l in avglosses]

    return x_axis, None, losses


def linefit(xaxis, yaxis, fit_range=None, out_range=None):
    import numpy as np
    if fit_range is not None:
        # asset xaxis increases
        start, end = 0, -1
        for i in range(len(xaxis)):
            if xaxis[i] <= fit_range[0] and ((i == len(xaxis) - 1) or xaxis[i + 1] > fit_range[0]):
                start = i    
            if xaxis[i] < fit_range[1] and ((i == len(xaxis) - 1) or xaxis[i + 1] >= fit_range[1]):
                end = i     
        if start == end:
            raise IndexError(f"{fit_range} out of range.")
        xaxis = xaxis[start: end]
        yaxis = yaxis[start: end]

    
    if out_range is None:
        out_range = fit_range
    outx = out_range

    z = np.polyfit(xaxis, yaxis, deg=1)
    return outx, [z[0] * _x + z[1] for _x in outx]


def draw_fig(data: list, xlim=(0, 301), ylim=(68, 84), xstep=None,ystep=None, save_path="./show.jpg"):
    assert isinstance(data[0], dict)
    from matplotlib import pyplot as plot
    fig, ax = plot.subplots(dpi=300, figsize=(24, 8))
    for d in data:
        length = min(len(d['x']), len(d['y']))
        x_axis = d['x'][:length]
        y_axis = d['y'][:length]
        label = d['label']
        ax.plot(x_axis, y_axis, label=label)
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


# =====================================

def main_vssm_():
    logpath = os.path.join(os.path.dirname(__file__), "../../logs")
    showpath = os.path.join(os.path.dirname(__file__), "./show/log")
    
    # baseline ===
    swin_tiny = f"{logpath}/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.json"
    swin_small = f"{logpath}/swin_small_224_b16x64_300e_imagenet_20210615_110219.json"
    swin_base = f"{logpath}/swin_base_224_b16x64_300e_imagenet_20210616_190742.json"
    convnext_baseline = f"{logpath}/convnext_modelarts-job-68076d57-44e0-4fa8-afac-cea5b1ef12f2-worker-0.log"

    # 2,2,9,2/2,2,27,2 no droppath but with linspace dropout ===
    vssmtiny_nodroppath = f"{logpath}/modelarts-job-bc7d1b2d-b288-42ba-83c7-047965cf08e0-worker-0.log"

    # 2,2,9,2/2,2,27,2 with droppath ===
    vssmtiny = f"{logpath}/modelarts-job-71acc542-bf6c-4731-a2ba-5eb710756bf9-worker-0.log"
    vsstiny_noconv = f"{logpath}/modelarts-job-6604f47e-feb0-4639-a41a-29b38472e2ce-worker-0.log"
    vssmbase = f"{logpath}/modelarts-job-98f7d241-31d3-45bb-87ea-85e3aa870895-worker-0.log"
    vssmsmall = f"{logpath}/modelarts-job-8f5c0423-03c2-4598-b460-6fe7e500ebb8-worker-0.log"
    vssmbasedrop06 = f"{logpath}/modelarts-job-da42b89f-6947-482b-9d77-286a76653402-worker-0.log"


    # 2,2,9,2/2,2,27,2 with droppath ===
    vssmdtiny = f"{logpath}/vssmdtinylog_rank0.txt"
    vssmdsmall = f"{logpath}/vssmdsmalllog_rank0.txt"
    vssmdbase = f"{logpath}/vssmdbaselog_rank0.txt"

    # =====================================================================
    x, accs, emaaccs = get_acc_mmpretrain(swin_tiny)
    lx, losses, avglosses = get_loss_mmpretrain(swin_tiny)
    swin_tiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_small)
    lx, losses, avglosses = get_loss_mmpretrain(swin_small)
    swin_small = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_base)
    lx, losses, avglosses = get_loss_mmpretrain(swin_base)
    swin_base = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_convnext(convnext_baseline)
    lx, losses, avglosses = get_loss_convnext(convnext_baseline)
    convnext_baseline = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmtiny_nodroppath)
    lx, losses, avglosses = get_loss_swin(vssmtiny_nodroppath, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmtiny_nodroppath = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vsstiny_noconv)
    lx, losses, avglosses = get_loss_swin(vsstiny_noconv, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vsstiny_noconv = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
    
    x, accs, emaaccs = get_acc_swin(vssmtiny)
    lx, losses, avglosses = get_loss_swin(vssmtiny, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmtiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmbase)
    lx, losses, avglosses = get_loss_swin(vssmbase, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmbase = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmsmall, split_ema=True)
    lx, losses, avglosses = get_loss_swin(vssmsmall, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmsmall = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmbasedrop06, split_ema=True)
    lx, losses, avglosses = get_loss_swin(vssmbasedrop06, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmbasedrop06 = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    # droppath + 2292 =======================================================
    fit_vssmbase = linefit(vssmsmall['xaxis'], vssmsmall['accs']['acc1'], fit_range=[100, 300], out_range=[60, 300])

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['xaxis'], y=swin_tiny['accs']['acc1'], label="swin_tiny"),
            dict(x=swin_small['xaxis'], y=swin_small['accs']['acc1'], label="swin_small"),
            dict(x=swin_base['xaxis'], y=swin_base['accs']['acc1'], label="swin_base"),
            # dict(x=convnext_baseline['xaxis'], y=convnext_baseline['accs']['acc1'], label="convnext_tiny_acc1_baseline"),
            # dict(x=convnext_baseline['xaxis'], y=convnext_baseline['emaaccs']['acc1'], label="convnext_tiny_acc1_ema_baseline"),
            # ======================================================================
            # ======================================================================
            dict(x=vssmtiny['xaxis'], y=vssmtiny['accs']['acc1'], label="vssmtiny"),
            # dict(x=vssmtiny_nodroppath['xaxis'], y=vssmtiny_nodroppath['accs']['acc1'], label="vssmtiny_nodroppath"),
            # dict(x=vsstiny_noconv['xaxis'], y=vsstiny_noconv['accs']['acc1'], label="vsstiny_noconv"),
            dict(x=vssmbase['xaxis'], y=vssmbase['accs']['acc1'], label="vssmbase"),
            dict(x=vssmsmall['xaxis'], y=vssmsmall['accs']['acc1'], label="vssmsmall"),
            dict(x=vssmsmall['xaxis'], y=vssmsmall['emaaccs']['acc1'], label="vssmsmall_ema"),
            dict(x=vssmbasedrop06['xaxis'], y=vssmbasedrop06['accs']['acc1'], label="vssmbasedrop06"),
            dict(x=vssmbasedrop06['xaxis'], y=vssmbasedrop06['emaaccs']['acc1'], label="vssmbasedrop06_ema"),
            # ======================================================================
            # dict(x=fit_vssmbase[0], y=fit_vssmbase[1], label="vssmbase"),
            # ======================================================================
        ], xlim=(20, 300), ylim=(70, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc_vssm.jpg")

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['loss_xaxis'], y=swin_tiny['avglosses'], label="swin_tiny"),
            dict(x=swin_small['loss_xaxis'], y=swin_small['avglosses'], label="swin_small"),
            dict(x=swin_base['loss_xaxis'], y=swin_base['avglosses'], label="swin_base"),
            # dict(x=convnext_baseline['loss_xaxis'], y=convnext_baseline['avglosses'], label="convnext_tiny_acc1_baseline"),
            # ======================================================================
            # dict(x=vssmtiny_nodroppath['loss_xaxis'], y=vssmtiny_nodroppath['avglosses'], label="vssmtiny_nodroppath"),
            # ======================================================================
            dict(x=vssmtiny['loss_xaxis'], y=vssmtiny['avglosses'], label="vssmtiny"),
            dict(x=vssmbase['loss_xaxis'], y=vssmbase['avglosses'], label="vssmbase"),
            dict(x=vssmsmall['loss_xaxis'], y=vssmsmall['avglosses'], label="vssmsmall"),
            dict(x=vssmbasedrop06['loss_xaxis'], y=vssmbasedrop06['avglosses'], label="vssmbasedrop06"),
            # ======================================================================
        ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss_vssm.jpg")

    # droppath + 2262 =======================================================

    x, accs, emaaccs = get_acc_swin(vssmdtiny, split_ema=True)
    lx, losses, avglosses = get_loss_swin(vssmdtiny, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmdtiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmdsmall, split_ema=True)
    lx, losses, avglosses = get_loss_swin(vssmdsmall, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmdsmall = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(vssmdbase, split_ema=True)
    lx, losses, avglosses = get_loss_swin(vssmdbase, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    vssmdbase = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['xaxis'], y=swin_tiny['accs']['acc1'], label="swin_tiny"),
            dict(x=swin_small['xaxis'], y=swin_small['accs']['acc1'], label="swin_small"),
            dict(x=swin_base['xaxis'], y=swin_base['accs']['acc1'], label="swin_base"),
            dict(x=vssmdtiny['xaxis'], y=vssmdtiny['accs']['acc1'], label="vssmdtiny"),
            dict(x=vssmdsmall['xaxis'], y=vssmdsmall['accs']['acc1'], label="vssmdsmall"),
            dict(x=vssmdbase['xaxis'], y=vssmdbase['accs']['acc1'], label="vssmdbase"),
            # ======================================================================
            dict(x=vssmdtiny['xaxis'], y=vssmdtiny['emaaccs']['acc1'], label="vssmdtiny_ema"),
            dict(x=vssmdsmall['xaxis'], y=vssmdsmall['emaaccs']['acc1'], label="vssmdsmall_ema"),
            dict(x=vssmdbase['xaxis'], y=vssmdbase['emaaccs']['acc1'], label="vssmdbase_ema"),
            # dict(x=fit_vssmbase[0], y=fit_vssmbase[1], label="vssmbase"),
            # ======================================================================
        ], xlim=(20, 300), ylim=(70, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc_vssmd.jpg")

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['loss_xaxis'], y=swin_tiny['avglosses'], label="swin_tiny"),
            dict(x=swin_small['loss_xaxis'], y=swin_small['avglosses'], label="swin_small"),
            dict(x=swin_base['loss_xaxis'], y=swin_base['avglosses'], label="swin_base"),
            # ======================================================================
            dict(x=vssmdtiny['loss_xaxis'], y=vssmdtiny['avglosses'], label="vssmdtiny"),
            dict(x=vssmdbase['loss_xaxis'], y=vssmdbase['avglosses'], label="vssmdbase"),
            dict(x=vssmdsmall['loss_xaxis'], y=vssmdsmall['avglosses'], label="vssmdsmall"),
            # ======================================================================
        ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss_vssmd.jpg")


def main_heat_():
    logpath = os.path.join(os.path.dirname(__file__), "../../logs")
    showpath = os.path.join(os.path.dirname(__file__), "./show/log")
    
    # baseline ===
    swin_tiny = f"{logpath}/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.json"
    swin_small = f"{logpath}/swin_small_224_b16x64_300e_imagenet_20210615_110219.json"
    swin_base = f"{logpath}/swin_base_224_b16x64_300e_imagenet_20210616_190742.json"
    convnext_baseline = f"{logpath}/convnext_modelarts-job-68076d57-44e0-4fa8-afac-cea5b1ef12f2-worker-0.log"

    heat_mini = "/home/LiuYue/Workspace/Visualize/output/heat_mini/rank0.log"
    heat_tiny = "/home/LiuYue/Workspace/Visualize/output/heat_tiny/default/log_rank0.txt"
    heat_base = "/home/LiuYue/Workspace/Visualize/output/heat_base/rank0.log"

    # =====================================================================
    x, accs, emaaccs = get_acc_mmpretrain(swin_tiny)
    lx, losses, avglosses = get_loss_mmpretrain(swin_tiny)
    swin_tiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_small)
    lx, losses, avglosses = get_loss_mmpretrain(swin_small)
    swin_small = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_base)
    lx, losses, avglosses = get_loss_mmpretrain(swin_base)
    swin_base = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_convnext(convnext_baseline)
    lx, losses, avglosses = get_loss_convnext(convnext_baseline)
    convnext_baseline = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(heat_mini, split_ema=True)
    lx, losses, avglosses = get_loss_swin(heat_mini, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    heat_mini = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(heat_tiny, split_ema=True)
    lx, losses, avglosses = get_loss_swin(heat_tiny, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    heat_tiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(heat_base, split_ema=True)
    lx, losses, avglosses = get_loss_swin(heat_base, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    heat_base = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['xaxis'], y=swin_tiny['accs']['acc1'], label="swin_tiny"),
            dict(x=swin_small['xaxis'], y=swin_small['accs']['acc1'], label="swin_small"),
            dict(x=swin_base['xaxis'], y=swin_base['accs']['acc1'], label="swin_base"),
            dict(x=heat_mini['xaxis'], y=heat_mini['accs']['acc1'], label="heat_mini"),
            dict(x=heat_tiny['xaxis'], y=heat_tiny['accs']['acc1'], label="heat_tiny"),
            dict(x=heat_base['xaxis'], y=heat_base['accs']['acc1'], label="heat_base"),
            # ======================================================================
            dict(x=heat_mini['xaxis'], y=heat_mini['emaaccs']['acc1'], label="heat_mini_ema"),
            dict(x=heat_tiny['xaxis'], y=heat_tiny['emaaccs']['acc1'], label="heat_tiny_ema"),
            dict(x=heat_base['xaxis'], y=heat_base['emaaccs']['acc1'], label="heat_base_ema"),
            # ======================================================================
        ], xlim=(30, 300), ylim=(60, 80), xstep=5, ystep=0.5, save_path=f"{showpath}/acc_heat.jpg")

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['loss_xaxis'], y=swin_tiny['avglosses'], label="swin_tiny"),
            dict(x=swin_small['loss_xaxis'], y=swin_small['avglosses'], label="swin_small"),
            dict(x=swin_base['loss_xaxis'], y=swin_base['avglosses'], label="swin_base"),
            # ======================================================================
            dict(x=heat_mini['loss_xaxis'], y=heat_mini['avglosses'], label="heat_mini"),
            dict(x=heat_tiny['loss_xaxis'], y=heat_tiny['avglosses'], label="heat_tiny"),
            dict(x=heat_base['loss_xaxis'], y=heat_base['avglosses'], label="heat_base"),
            # ======================================================================
        ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss_heat.jpg")


def main_heatwzz():
    logpath = os.path.join(os.path.dirname(__file__), "../../logs")
    showpath = os.path.join(os.path.dirname(__file__), "./show/log")
    
    # baseline ===
    swin_tiny = f"{logpath}/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.json"
    swin_small = f"{logpath}/swin_small_224_b16x64_300e_imagenet_20210615_110219.json"
    swin_base = f"{logpath}/swin_base_224_b16x64_300e_imagenet_20210616_190742.json"
    convnext_baseline = f"{logpath}/convnext_modelarts-job-68076d57-44e0-4fa8-afac-cea5b1ef12f2-worker-0.log"

    heat_base_d005 = "/home/LiuYue/Workspace/Visualize/output/wzz/vit_heat_noshift_base_224_14layers_clipgrad5.0.txt"
    heat_base_d01 = "/home/LiuYue/Workspace/Visualize/output/wzz/vit_heat_noshift_base_224_14layers_dp0.1.txt"

    # =====================================================================
    x, accs, emaaccs = get_acc_mmpretrain(swin_tiny)
    lx, losses, avglosses = get_loss_mmpretrain(swin_tiny)
    swin_tiny = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_small)
    lx, losses, avglosses = get_loss_mmpretrain(swin_small)
    swin_small = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_mmpretrain(swin_base)
    lx, losses, avglosses = get_loss_mmpretrain(swin_base)
    swin_base = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_convnext(convnext_baseline)
    lx, losses, avglosses = get_loss_convnext(convnext_baseline)
    convnext_baseline = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(heat_base_d005, split_ema=False)
    lx, losses, avglosses = get_loss_swin(heat_base_d005, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    heat_base_d005 = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    x, accs, emaaccs = get_acc_swin(heat_base_d01, split_ema=False)
    lx, losses, avglosses = get_loss_swin(heat_base_d01, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
    heat_base_d01 = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)

    fit_heatbase = linefit(heat_base_d005['xaxis'], heat_base_d005['accs']['acc1'], fit_range=[100, 300], out_range=[60, 300])

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['xaxis'], y=swin_tiny['accs']['acc1'], label="swin_tiny"),
            dict(x=swin_small['xaxis'], y=swin_small['accs']['acc1'], label="swin_small"),
            dict(x=swin_base['xaxis'], y=swin_base['accs']['acc1'], label="swin_base"),
            dict(x=heat_base_d005['xaxis'], y=heat_base_d005['accs']['acc1'], label="heat_base_d005"),
            dict(x=heat_base_d01['xaxis'], y=heat_base_d01['accs']['acc1'], label="heat_base_d01"),
            dict(x=fit_heatbase[0], y=fit_heatbase[1], label="fit_heatbase"),
            # ======================================================================
        ], xlim=(30, 300), ylim=(60, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc_heatwzz.jpg")

    if True:
        draw_fig(data=[
            dict(x=swin_tiny['loss_xaxis'], y=swin_tiny['avglosses'], label="swin_tiny"),
            dict(x=swin_small['loss_xaxis'], y=swin_small['avglosses'], label="swin_small"),
            dict(x=swin_base['loss_xaxis'], y=swin_base['avglosses'], label="swin_base"),
            # ======================================================================
            dict(x=heat_base_d005['loss_xaxis'], y=heat_base_d005['avglosses'], label="heat_base_d005"),
            dict(x=heat_base_d01['loss_xaxis'], y=heat_base_d01['avglosses'], label="heat_base_d01"),
            # ======================================================================
        ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss_heatwzz.jpg")


# =====================================
def main_vssm():
    results = {}
    logpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssm/")
    showpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssm/fig")

    files = dict(
        ti = f"{logpath}/vssmtiny/vmamba_tiny_dp01_e292_woema.log",
        sm = f"{logpath}/vssmsmall/vmamba_small_dp03_e238_ema.log",
        ba = f"{logpath}/vssmbase/vmamba_base_dp06_e241_ema.log",
        ba05 = f"{logpath}/vssmbasedp05/vmamba_base_dp05_e260_woema.log",
    )
    
    for name, file in files.items():
        x, accs, emaaccs = get_acc_swin(file, split_ema=True)
        lx, losses, avglosses = get_loss_swin(file, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
        file = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
        results.update({name: file})

    draw_fig(data=[
        *[dict(x=file['xaxis'], y=file['accs']['acc1'], label=name) for name, file in results.items()], 
        *[dict(x=file['xaxis'], y=file['emaaccs']['acc1'], label=f"{name}_ema") for name, file in results.items()],
    ], xlim=(30, 300), ylim=(60, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc.jpg")

    draw_fig(data=[
        *[dict(x=file['loss_xaxis'], y=file['avglosses'], label=name)  for name, file in results.items()],
    ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss.jpg")
    

def main_vssmd():
    results = {}
    logpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssmd/")
    showpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssmd/fig")

    files = dict(
        ti = f"{logpath}/vssmdtiny/default/log_rank0.txt",
        sm = f"{logpath}/vssmdsmall/default/log_rank0.txt",
        ba = f"{logpath}/vssmdbase/default/log_rank0.txt",
    )
    
    for name, file in files.items():
        x, accs, emaaccs = get_acc_swin(file, split_ema=True)
        lx, losses, avglosses = get_loss_swin(file, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
        file = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
        results.update({name: file})

    draw_fig(data=[
        *[dict(x=file['xaxis'], y=file['accs']['acc1'], label=name) for name, file in results.items()], 
        *[dict(x=file['xaxis'], y=file['emaaccs']['acc1'], label=f"{name}_ema") for name, file in results.items()],
    ], xlim=(30, 300), ylim=(60, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc.jpg")

    draw_fig(data=[
        *[dict(x=file['loss_xaxis'], y=file['avglosses'], label=name)  for name, file in results.items()],
    ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss.jpg")
    

def main_heat():
    results = {}
    logpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/heat/")
    showpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/heat/fig")

    files = dict(
        mi = f"{logpath}/heat_mini/log_rank0.txt",
        ti = f"{logpath}/heat_tiny/log_rank0.txt",
        sm = f"{logpath}/heat_small/log_rank0.txt",
        ba = f"{logpath}/heat_base/log_rank0.txt",
        tiso = f"{logpath}/heat_tiny_softmax/log_rank0.txt",
    )
    
    for name, file in files.items():
        x, accs, emaaccs = get_acc_swin(file, split_ema=True)
        lx, losses, avglosses = get_loss_swin(file, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
        file = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
        results.update({name: file})

    draw_fig(data=[
        *[dict(x=file['xaxis'], y=file['accs']['acc1'], label=name) for name, file in results.items()], 
        *[dict(x=file['xaxis'], y=file['emaaccs']['acc1'], label=f"{name}_ema") for name, file in results.items()],
    ], xlim=(30, 300), ylim=(60, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc.jpg")

    draw_fig(data=[
        *[dict(x=file['loss_xaxis'], y=file['avglosses'], label=name)  for name, file in results.items()],
    ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss.jpg")
    

def main_vssm1():
    results = {}
    logpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssm1/")
    showpath = os.path.join(os.path.dirname(__file__), "../../ckpts/private/classification/vssm1/vssm1tifig")

    files = dict(
        # ti1 = f"{logpath}/vssm_tiny_224_1.log",
        # ti1v1 = f"{logpath}/vssm_tiny_224_1v1.log",
        # ti0211 = f"{logpath}/vssm_tiny_224_0211.log",
        # ti0211v1 = f"{logpath}/vssm_tiny_224_0211v1.log",
        # ti0212 = f"{logpath}/vssm_tiny_224_0212.log",
        # ti0213 = f"{logpath}/vssm_tiny_224_0213.log",
        # ti0215 = f"{logpath}/vssm_tiny_224_0215.log",
        # ti0216 = f"{logpath}/vssm_tiny_224_0216.log",
        # ti0217 = f"{logpath}/vssm_tiny_224_0217.log",
        # ti0218 = f"{logpath}/vssm_tiny_224_0218.log",
        # ti0219 = f"{logpath}/vssm_tiny_224_0219.log",
        # ti0220 = f"{logpath}/vssm_tiny_224_0220.log",
        # ti0221 = f"{logpath}/vssm_tiny_224_0221.log",
        # ti0222 = f"{logpath}/vssm_tiny_224_0222.log",
        # ti0223 = f"{logpath}/vssm_tiny_224_0223.log",
        # ti0225 = f"{logpath}/vssm_tiny_224_0225.log",
        # ti0229 = f"{logpath}/vssm_tiny_224_0229.log",
        ti0230 = f"/home/LiuYue/Workspace/PylanceAware/ckpts/private/classification/vssm1/Collect/classification/vssm1/vssm1_tiny_0230/log_rank0.txt",
        # ti0309 = f"/home/LiuYue/TmpMount/DataHPC8/nodeHPC8/out/vssm1_tiny_0309/20240308171406/log_rank0.txt",
        # ti0310 = f"/home/LiuYue/TmpMount/DataHPC9/nodeHPC9/out/vssm1_tiny_0310/20240309044533/log_rank0.txt",
        # ti0311 = f"/home/LiuYue/TmpMount/DataHPC8/nodeHPC8/out/vssm1_tiny_0311/20240310033609/log_rank0.txt",
        # ti0312 = f"/home/LiuYue/Workspace/PylanceAware/ckpts/private/classification/vssm1/Collect/classification/vssm1/vssm1_tiny_0312/log_rank0.txt",
        ti0313 = f"/home/LiuYue/Workspace/PylanceAware/ckpts/private/classification/vssm1/Collect/classification/vssm1/vssm1_tiny_0313/log_rank0.txt",
        # ti0314 = f"/home/LiuYue/TmpMount/DataHPC9/nodeHPC9/out/vssm1_tiny_0314/20240312030720/log_rank0.txt",
        # ti0315 = f"/home/LiuYue/Workspace/PylanceAware/ckpts/private/classification/vssm1/Collect/classification/vssm1/vssm1_tiny_0315/log_rank0.txt",
        # ti0316 = f"/home/LiuYue/Workspace/PylanceAware/ckpts/private/classification/vssm1/Collect/classification/vssm1/vssm1_tiny_0316/log_rank0.txt",
        # ti0317 = f"/home/LiuYue/TmpMount/DataHPC9/nodeHPC9/out/vssm1_tiny_0317/20240314160739/log_rank0.txt",
        # ti0318 = f"/home/LiuYue/TmpMount/DataHPC8/nodeHPC8/out/vssm1_tiny_0318/20240314161620/log_rank0.txt",
        ti03182 = f"/home/LiuYue/TmpMount/DataHPC8/nodeHPC8/out/vssm1_tiny_03182/20240316020121/log_rank0.txt",
        ti0319 = f"/home/LiuYue/TmpMount/DataHPC9/nodeHPC9/out/vssm1_tiny_0319/20240316022352/log_rank0.txt",
        ti0322 = f"/home/LiuYue/TmpMount/DataHPC9/nodeHPC9/out/vssm1_tiny_0322/20240317142359/log_rank0.txt",
        ti0323 = f"/home/LiuYue/TmpMount/DataHPC8/nodeHPC8/out/vssm1_tiny_0323/20240317150405/log_rank0.txt",
    )
    
    for name, file in files.items():
        x, accs, emaaccs = get_acc_swin(file, split_ema=True)
        lx, losses, avglosses = get_loss_swin(file, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
        file = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
        results.update({name: file})

    draw_fig(data=[
        *[dict(x=file['xaxis'], y=file['accs']['acc1'], label=name) for name, file in results.items()], 
        *[dict(x=file['xaxis'], y=file['emaaccs']['acc1'], label=f"{name}_ema") for name, file in results.items()],
    ], xlim=(30, 300), ylim=(60, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc.jpg")

    draw_fig(data=[
        *[dict(x=file['loss_xaxis'], y=file['avglosses'], label=name)  for name, file in results.items()],
    ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss.jpg")


if __name__ == "__main__":
    ...
    # main_vssm()
    # main_vssmd()
    # main_heat()
    main_vssm1()

