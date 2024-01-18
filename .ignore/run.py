import os
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0")
    parser.add_argument("--train_url", type=str, default="/home/ma-user/modelarts/outputs/train_url_0")
    parser.add_argument("--base_url", type=str, default="/home/ma-user/modelarts")
    parser.add_argument("--job_url", type=str, default="/home/ma-user/modelarts/user-job-dir/vmamba")    
    # =======================================
    parser.add_argument("--code_url", type=str, default="Swin-Transformer-main")
    parser.add_argument("--pack_url", type=str, default="package")
    parser.add_argument("--conda_url", type=str, default="miniconda3") # no need to change
    parser.add_argument("--conda_env", type=str, default="mamba") # no need to change
    # =======================================
    parser.add_argument("--init_method", type=str, default="tcp://127.0.0.1:6666")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--nprocs", type=int, default=8)
    # =======================================
    parser.add_argument("--pycmds", type=str, default="main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 64 --data-path /home/ma-user/modelarts/inputs/data_url_0 --output /home/ma-user/modelarts/outputs/train_url_0 --model_ema --opts MODEL.TYPE vssm MODEL.NAME vssmtiny ")
    args = parser.parse_args()
    return args


def create_env(pack_url="/packages", conda_url="/miniconda3", conda_env="mamba"):
    # set path first to confirm use newly installed conda
    PATH = f"{conda_url}/envs/{conda_env}/bin:{conda_url}/bin:{os.environ['PATH']}"
    os.environ["PATH"] = PATH

    # can not connect to web !!!!!
    cmdsenv = [
        f"unset PYTHONPATH; bash {pack_url}/Miniconda3* -bfup {conda_url}",
        # f"conda create -n {conda_env} python==3.10 -y",
        f"conda create -n {conda_env} --clone base",
        "python -VV",
        "pip -V",
        # "pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117",
        "pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0",
        "pip install packaging",
        "pip install timm==0.4.12",
        "pip install pytest chardet yacs termcolor",
        "pip install submitit tensorboardX",
        "pip install triton==2.0.0",
        f"pip install {pack_url}/causal_conv1d*",
        f"pip install {pack_url}/mamba_ssm*",
    ]

    if not os.path.exists(f"{conda_url}/envs/{conda_env}/bin"):
        for cmd in cmdsenv:
            print("=" * 32, cmd, "=" * 32, flush=True)
            os.system(cmd)
    
    return PATH


def copy_dataset():
    # mox.file.copy_parallel()
    os.system("tar -xf /cache/imagenet.tar -C /cache/")


def run_code(
    PATH=os.environ["PATH"], 
    maddr="127.0.0.1", 
    mport="12345", 
    nrank="0", 
    nnodes="1", 
    nprocs="8", 
    log_url="./log", 
    code_url=".",
    pycmds="main.py",
):
    os.environ["PATH"] = PATH
    os.environ["MASTER_ADDR"] = str(maddr)
    os.environ["MASTER_PORT"] = str(mport)
    os.environ["NODE_RANK"] = str(nrank)
    os.environ["NNODES"] = str(nnodes)
    os.environ["NPROC_PER_NODE"] = str(nprocs)
    os.environ["LOG_DIR"] = log_url
    os.makedirs(log_url, exist_ok=True)

    cmdsrun = [
        "unset PYTHONPATH",
        "printenv",
        "python -VV",
        "pip -V",
        "nvcc --version",
        "pip freeze",
        f"cd {code_url}; python -m torch.distributed.launch --nnodes {nnodes} --node_rank {nrank} --nproc_per_node {nprocs} --master_addr {maddr} --master_port {mport} --log_dir {log_url} {pycmds}",
    ]

    for cmd in cmdsrun:
        print("=" * 32, cmd, "=" * 32, flush=True)
        os.system(cmd)


# =========================================
args = get_args()
print("=" * 32, "args", "=" * 32, flush=True)
print(args)
print("=" * 32, "ls / -alh", "=" * 32, flush=True)
os.system("ls / -alh")
print("=" * 32, f"tree {os.environ['HOME']} -L 2", "=" * 32, flush=True)
os.system(f"tree {os.environ['HOME']} -L 2")
print("=" * 32, f"tree {args.base_url} -L 2", "=" * 32, flush=True)
os.system(f"tree {args.base_url} -L 2")
print("=" * 32, f"tree {args.train_url} -L 2", "=" * 32, flush=True)
os.system(f"tree {args.train_url} -L 2")
print("=" * 32, f"tree {args.data_url} -L 2", "=" * 32, flush=True)
os.system(f"tree {args.data_url} -L 2")
print("=" * 32, f"ls {args.job_url} -alh", "=" * 32, flush=True)
os.system(f"ls {args.job_url} -alh")
time.sleep(10)

PATH = create_env(
    pack_url=os.path.join(args.job_url, args.pack_url), 
    conda_url=os.path.join(args.job_url, args.conda_url), 
    conda_env=args.conda_env,
)

run_code(
    PATH=PATH,
    maddr=args.init_method[:-(len(args.init_method.split(":")[-1]) + 1)][len("tcp://"):],
    mport=args.init_method.split(":")[-1],
    nrank=args.rank,
    nnodes=args.world_size,
    nprocs=args.nprocs,
    log_url=os.path.join(args.train_url, "logs", time.strftime("%Y%m%d_%H%M%S")),
    code_url=os.path.join(args.job_url, args.code_url),
    pycmds=args.pycmds.strip("\""),
)


"""
python ${MA_JOB_DIR}/vmamba/run.py
# AUTO PARAMS ========
# --data_url=/home/ma-user/modelarts/inputs/data_url_0
# --train_url=/home/ma-user/modelarts/outputs/train_url_0
# --init_method "tcp://$(echo ${VC_WORKER_HOSTS} | cut -d "," -f 1):6666" --rank ${VC_TASK_INDEX} --world_size 2
# SELF PARAMS ========
# --nprocs 8
# --base_url /home/ma-user/modelarts
# --job_url /home/ma-user/modelarts/user-job-dir/vmamba
# --code_url classification
# --pack_url package
# --pycmds "main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 64 --data-path /home/ma-user/modelarts/inputs/data_url_0/imagenet --output /home/ma-user/modelarts/outputs/train_url_0/ --opts MODEL.TYPE vssm MODEL.SWIN.DEPTHS  MODEL.NAME vssmtiny"
$(date +%Y%m%d%H%M%S)
"""



