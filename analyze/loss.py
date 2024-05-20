import os
import shutil
import torch
import numpy as np


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
    fig, ax = plot.subplots(dpi=400, figsize=(24, 8))
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


def readlog_classification(logfile):
    _logs = open(logfile).readlines()

    MAX_SEARCH = 300
    _epochs, _accs, _emaaccs = [], [], []
    for i in range(0, len(_logs)):
        _lr = _logs[i]
        if "INFO" in _lr and f"ckpt_epoch_" in _lr and f".pth saved !!!\n" in _lr:
            epoch = int(_lr.split("ckpt_epoch_")[1].split(".pth")[0])
            _acc, _emaacc = -1, -1
            for j in range(i + 1, min(i + MAX_SEARCH, len(_logs))):
                if f"INFO Max accuracy:" in _logs[j]:
                    assert "INFO Accuracy of the network" in _logs[j-1]
                    assert "INFO  * Acc@1" in _logs[j-2]
                    _acc = float(_logs[j-2].split("INFO  * Acc@1")[1].strip().split(" ")[0].strip())
                if f"INFO Max accuracy ema:" in _logs[j]:
                    assert "INFO Accuracy of the network" in _logs[j-1]
                    assert "INFO  * Acc@1" in _logs[j-2]
                    _emaacc = float(_logs[j-2].split("INFO  * Acc@1")[1].strip().split(" ")[0].strip())
                if f"INFO Train:" in _logs[j]:
                    break
            _epochs.append(epoch)
            _accs.append(_acc)
            _emaaccs.append(_emaacc)
    
    _max_acc = np.array(_accs).max() if len(_accs) > 0 else -1
    _max_acc_idx = np.flatnonzero(np.array(_accs)  == _max_acc)
    _max_emaacc = np.array(_emaaccs).max() if len(_emaaccs) > 0 else -1
    _max_emaacc_idx = np.flatnonzero(np.array(_emaaccs)  == _max_emaacc)

    _mkidx = np.union1d(_max_acc_idx, _max_emaacc_idx)
    _final_epoch = max(_epochs)
    print(f"\033[4;32mmax acc ema: {_max_emaacc}, {[_epochs[i] for i in _max_emaacc_idx]}; max acc: {_max_acc}, {[_epochs[i] for i in _max_acc_idx]}; final ckpt: {_final_epoch}; \033[0m")
    _ckpts = [f"ckpt_epoch_{e}.pth" for e in set([_final_epoch, *[_epochs[i] for i in _mkidx]]) if e != -1]

    return _ckpts


def readlog_mmdetection(logfile):
    _logs = open(logfile).readlines()

    _coco_bbox_mAPs, _coco_segm_mAPs, _epochs, _keylogs = [], [], [], []
    for i, _l in enumerate(_logs):
        if ("Epoch(val)" in _l) and (" eta: 0:" not in _l):
            epoch = int(_l.split("Epoch(val) [")[1].split("][")[0].strip())
            _epochs.append(epoch)
            if "coco/bbox_mAP" in _l:
                _coco_bbox_mAPs.append(float(_l.split(" coco/bbox_mAP: ")[1].strip().split(" ")[0].strip()))
            if "coco/segm_mAP" in _l:
                _coco_segm_mAPs.append(float(_l.split(" coco/segm_mAP: ")[1].strip().split(" ")[0].strip()))
            _keylogs.append(_l.strip("\n"))
    
    _max_coco_bbox_mAP = np.array(_coco_bbox_mAPs).max() if len(_coco_bbox_mAPs) > 0 else -1
    _max_bbox_idx = np.flatnonzero(_coco_bbox_mAPs == _max_coco_bbox_mAP)
    _max_coco_segm_mAP = np.array(_coco_segm_mAPs).max() if len(_coco_segm_mAPs) > 0 else -1
    _max_segm_idx = np.flatnonzero(_coco_segm_mAPs == _max_coco_segm_mAP)

    _mkidx = np.union1d(_max_bbox_idx, _max_segm_idx)
    _mkepochs = [_epochs[i] for i in _mkidx]
    _mklogs = [_keylogs[i] for i in _mkidx]
    _final_epoch = max(_epochs)

    print(f"\033[4;32mbboxmAP: {_max_coco_bbox_mAP}; segmmAP: {_max_coco_segm_mAP}; _mkepochs: {_mkepochs}; final epoch: {_final_epoch}; \033[0m")
    for l in _mklogs:
        perfs = " ".join([
            l.split("Epoch(val) ")[1].split(" ")[0],
            l.split("coco/bbox_mAP: ")[1].split(" ")[0],
            l.split("coco/bbox_mAP_50: ")[1].split(" ")[0],
            l.split("coco/bbox_mAP_75: ")[1].split(" ")[0],
            l.split("coco/segm_mAP: ")[1].split(" ")[0],
            l.split("coco/segm_mAP_50: ")[1].split(" ")[0],
            l.split("coco/segm_mAP_75: ")[1].split(" ")[0],
        ])
        print(f"\033[4;32m{perfs} \033[0m")
    print(_mklogs, logfile)
    _ckpts = [f"epoch_{e}.pth" for e in set([_final_epoch, *_mkepochs]) if e != -1]

    return _ckpts


def readlog_mmsegmentation(logfile, test=False):
    _logs = open(logfile).readlines()

    _mious, _iters, _keylogs = [], [], []
    for i, _l in enumerate(_logs):
        if test:
            if ("INFO - Iter(test) [" in _l) and (" eta: " not in _l):
                _iters.append(-1)
                _mious.append(float(_l.split(" mIoU:")[1].strip().split(" ")[0].strip()))
                _keylogs.append(_l.strip("\n"))
        else:
            if ("INFO - Iter(val) [" in _l) and (" eta: " not in _l):
                _iter = None
                for j in range(i, -1000, -1):
                    if "Saving checkpoint at" in _logs[j]:
                        _iter = int(_logs[j].split("Saving checkpoint at ")[1].strip().split(" ")[0].strip())
                        break
                assert isinstance(_iter, int), "ERROR: can not find iter"
                _iters.append(_iter)
                _mious.append(float(_l.split(" mIoU:")[1].strip().split(" ")[0].strip()))
                _keylogs.append(_l.strip("\n"))

    _max_miou = np.array(_mious).max() if len(_mious) > 0 else -1
    _max_miou_idx = np.flatnonzero(_mious == _max_miou)

    _mkidx = _max_miou_idx
    _mkiters = [_iters[i] for i in _mkidx]
    _mklogs = [_keylogs[i] for i in _mkidx]
    _final_iter = max(_iters)

    print(f"\033[4;32mmiou: {_max_miou}; _mkiters: {_mkiters}; final iter: {_final_iter}; \033[0m")
    print(_mklogs, logfile)
    _ckpts = [f"iter_{e}.pth" for e in set([_final_iter, *_mkiters]) if e != -1]

    return _ckpts


def cpclassification(src, name, dstpath="", fake_copy=False, update=False, onlylog=False):
    dst = os.path.join(dstpath, name)
    os.makedirs(dst, exist_ok=True)
    print(f"\033[4;32m{name} =======================================\033[0m")

    for file in ["config.json", "log_rank0.txt"]:
        if os.path.exists(os.path.join(dst, file)):
            print(f"WARNING: file [{os.path.join(dst, file)}] exist already")
            if not update:
                continue
        _s = os.path.join(src, file)
        assert os.path.exists(dst) and os.path.isdir(dst)
        assert os.path.exists(_s), f"Not found: {_s}"
        print(f"copy from [{_s}] to [{dst}]")
        if not fake_copy:
            shutil.copy(_s, dst)

    _ckpts = readlog_classification(os.path.join(os.path.abspath(dst), "log_rank0.txt"))

    if not onlylog:
        for file in _ckpts:
            _s = os.path.join(src, file)
            assert os.path.exists(_s)
            if os.path.exists(os.path.join(dst, file)):
                print(f"WARNING: file [{os.path.join(dst, file)}] exist already")
                with open(os.path.join(dst, file), "rb") as f:
                    torch.load(f)
            else:
                assert os.path.exists(dst) and os.path.isdir(dst)
                print(f"copy from [{_s}] to [{dst}]")
                if not fake_copy:
                    shutil.copy(_s, dst)


def puremodel(ickptfile=".", opath=".", key="model", convert_key="model", name="vssmtmp"):
    ckptname = os.path.basename(ickptfile)
    ilogfile = os.path.join(os.path.dirname(ickptfile), "log_rank0.txt")
    opath = os.path.join(opath, name)
    ockptfile = os.path.join(opath, f"{name}_{ckptname}")
    ologfile = os.path.join(opath, f"{name}.txt")
    os.makedirs(opath, exist_ok=True)
    
    print(f"{name} =======================================")
    _ckpts = readlog_classification(ilogfile)
    
    _ckpt = torch.load(open(ickptfile, "rb"), map_location=torch.device("cpu"))
    if key not in _ckpt.keys():
        raise KeyError(f"key {key} not in ckpt.keys: {_ckpt.keys()}")
    _ockpt = {convert_key: _ckpt[key]}
    if os.path.exists(ockptfile):
        print(f"WARNING file {ockptfile} exists.")
    else:
        torch.save(_ockpt, open(ockptfile, "wb"))
        print(f"{ockptfile} saved...")

    assert os.path.exists(ilogfile), f"log file {ilogfile} not found"
    if os.path.exists(ologfile):
        print(f"WARNING file {ologfile} exists.")
    else:
        shutil.copy(ilogfile, ologfile)
        print(f"{ologfile} saved...")


def puremodelmmdet(ilogfile, opath=".", fake_copy=False, mode="coco"):
    ilogfile = os.path.abspath(ilogfile)
    ilogfiledir = os.path.dirname(ilogfile)
    ipath = os.path.dirname(ilogfiledir)
    name = os.path.basename(ipath)
    configfile = os.path.join(ipath, f"{name}.py")
    assert os.path.exists(configfile), f"can not process directory: {os.listdir(ipath)}"
    dst = os.path.join(opath, name)
    ologfile = os.path.join(dst, f"{name}.log")
    os.makedirs(dst, exist_ok=True)
    print(f"\033[4;32m{name} =======================================\033[0m")

    for _s in [ilogfiledir, configfile]:
        _o = os.path.join(dst, os.path.basename(_s))
        if os.path.exists(_o):
            print(f"WARNING: file [{_o}] exist already")
        else:
            assert os.path.exists(dst) and os.path.isdir(dst)
            print(f"copy from [{_s}] to [{dst}]")
            if not fake_copy:
                shutil.copytree(_s, _o) if os.path.isdir(_s) else shutil.copy(_s, dst)

    assert os.path.exists(ilogfile), f"log file {ilogfile} not found"
    if os.path.exists(ologfile):
        print(f"WARNING file {ologfile} exists.")
    else:
        shutil.copy(ilogfile, ologfile)
        print(f"{ologfile} saved...")

    if mode in ["coco"]:
        _ckpts = readlog_mmdetection(ilogfile)
    elif mode in ["ade20k"]:
        _ckpts = readlog_mmsegmentation(ilogfile)
    
    for _s in _ckpts:
        ickptfile = os.path.join(ipath, _s)
        ockptfile = os.path.join(dst, f"{name}_{_s}")
        _ckpt = torch.load(open(ickptfile, "rb"), map_location=torch.device("cpu"))
        _ockpt = {"meta": _ckpt["meta"], "state_dict": _ckpt["state_dict"]}
        if os.path.exists(ockptfile):
            print(f"WARNING file {ockptfile} exists.")
        else:
            torch.save(_ockpt, open(ockptfile, "wb"))
            print(f"{ockptfile} saved...")


def main_vssm():
    results = {}
    showpath = os.path.join(os.path.dirname(__file__), "./show/vssm1tifig")

    files = dict(
    )
    
    for name, file in files.items():
        x, accs, emaaccs = get_acc_swin(file, split_ema=True)
        lx, losses, avglosses = get_loss_swin(file, x1e=torch.tensor(list(range(0, 1251, 10))).view(1, -1) / 1251, scale=1)
        file = dict(xaxis=x, accs=accs, emaaccs=emaaccs, loss_xaxis=lx, losses=losses, avglosses=avglosses)
        results.update({name: file})

    draw_fig(data=[
        *[dict(x=file['xaxis'], y=file['accs']['acc1'], label=name) for name, file in results.items()], 
        *[dict(x=file['xaxis'], y=file['emaaccs']['acc1'], label=f"{name}_ema") for name, file in results.items()],
    ], xlim=(30, 300), ylim=(70, 85), xstep=5, ystep=0.5, save_path=f"{showpath}/acc.jpg")

    draw_fig(data=[
        *[dict(x=file['loss_xaxis'], y=file['avglosses'], label=name)  for name, file in results.items()],
    ], xlim=(10, 300), ylim=(2,5), save_path=f"{showpath}/loss.jpg")


if __name__ == "__main__":
    main_vssm()
    
