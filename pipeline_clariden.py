#!/usr/bin/env python3
import os, glob
from pathlib import Path
os.environ.pop("SQUEUE_FORMAT", None)
from simple_slurm import Slurm


# Equivalent to workflow.launchDir
LAUNCH_DIR = Path(os.getcwd())
LOG_DIR = LAUNCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def array_string(ids, max_running=None):

    array_arg = ",".join(map(str, ids)) 
    if max_running is not None:
        array_arg += f"%{max_running}"
    return array_arg

def add_environment_variables(command):

    command = f"""
    export WANDB_API_KEY=$(cat {os.path.expanduser('~/.secrets/wandb-api-key')}); \\
    export WANDB_RUN_GROUP="slurm-$SLURM_ARRAY_JOB_ID"; \\
    """ + command
    return command

def get_logs(name):

    return {"output": str(LOG_DIR / f"{name}.%A_%a.out"), "error": str(LOG_DIR / f"{name}.%A_%a.err")}


def clear_logs(name):

    log_dict = get_logs(name)
    logs_stdout = glob.glob(log_dict["output"].replace("%A_%a", "*"))
    logs_stderr = glob.glob(log_dict["error"].replace("%A_%a", "*"))
    n_removed = 0
    for log in logs_stdout + logs_stderr:
        if os.path.exists(log):
            os.remove(log)  
            n_removed += 1
    print(f'Removed {n_removed} logs')

def submit_job(slurm_args, name, command):

    slurm_args["job_name"] = name
    slurm_args.update(get_logs(name))
    slurm = Slurm(**slurm_args)

    print(f'----------------- Submitting {name}')
    print(f'Clearing logs {name}')
    clear_logs(name)
    print(f'Command: {command}')
    job_id = slurm.sbatch(command)
    print(f"Submitted {name} jobid: {job_id}")
    return job_id


########################################################################################
########################################################################################
##
##
## Jobs submission definitions
##
##
########################################################################################
########################################################################################

def submit_paramtables(
    *,
    name,
    config,
    dir_out,
    **kwargs,
):
    if name.startswith("//"):  return None

    slurm_args = {
        "cpus_per_task": 1,
        "mem": "15600M",
        "time": "1:00:00",
        "partition": "performance",
        "clusters": "cluster",
        "array": [0],
    } | kwargs
    
    command = f"""
    pixi run uv run python -m cosmogridv11.apps.run_paramtables shell_permutations \\
        --config="{LAUNCH_DIR / config}" \\
        --dir_out="{LAUNCH_DIR / dir_out}" \\
        --verbosity=info
    """
        
    job_id = submit_job(slurm_args, name, command)
    return job_id



def submit_probemaps(
    *,
    name,
    config,
    dir_out,
    **kwargs,
):
    if name.startswith("//"):  return None

    slurm_args = {
        "cpus_per_task": 8,
        "mem_per_cpu": "1950M",
        "time": "12:00:00",
        "partition": "cpu-daily",
        "clusters": "calc-cpu",
    } | kwargs
    
    command = f"""
    pixi run uv run python -m cosmogridv11.apps.run_probemaps main \\
        --config="{LAUNCH_DIR / config}" \\
        --dir_out="{LAUNCH_DIR / dir_out}" \\
        --num_maps_per_index=10 \\
        --indices="$SLURM_ARRAY_TASK_ID" \\
        --verbosity=info
    """
    
    job_id = submit_job(slurm_args, name, command)
    return job_id

def submit_postprocessing(
    *,
    name,
    config,
    dir_in,
    dir_out,
    profile=False,
    submit=False,
    **kwargs,
):

    if name.startswith("//"):  return None

    slurm_args = {
        "cpus_per_task": 6,
        "mem_per_cpu": "1950M",
        "time": "6:00:00",
        "partition": "cpu-daily",
        "clusters": "calc-cpu",
    } | kwargs
    
    command = f"""
    pixi run uv run {f"mprof run --interval 0.02 " if profile else ""} python -m msfm.apps.run_onthefly_postprocessing wds \\
        --config="{LAUNCH_DIR / config}" \\
        --dir_in={LAUNCH_DIR / dir_in} \\
        --dir_out={LAUNCH_DIR / dir_out} \\
        --cosmogrid_version="1.1" \\
        --indices={"$SLURM_ARRAY_TASK_ID" if submit else "0"} \\
        --verbosity={"debug" if profile else "info"} \\
        --max_sleep={1 if profile else 120}
    """
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None
    return job_id




def execute_training_parallel(
    *,
    name,
    submit=False,
    test=True,
    array=None,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = {
        # "cpus_per_task": 96,
        # "mem_per_cpu": "1950M",
        "nodes": 1,
        "exclusive": True,
        "mem_per_cpu": 2900,
        "cpus_per_task": 288,
        "time": "0:30:00",
        "partition": "debug",
        "gres": "gpu:4",
        "account": "a0186",
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    # srun --time=2:0:0 -n1 -c32 --mem-per-cpu=1950 --gpus-per-task=1 -A a0186 --mpi=pmix --network=disable_rdzv_get --environment=./edf.toml --pty bash -c "cd $prev_home; exec bash --rcfile /capstor/scratch/cscs/tomaszk/.bashrc -i"
    command = f"""
    srun --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
            bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
            uv run  \\
            torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
            -m euclid_multiprobe_deeplss_training.cli \\
            --config="{LAUNCH_DIR}/config_deeplss.yaml" \\
            --verbosity=info \\
            train \\
            --tag={name} \\
            --wandb-mode={"online" if submit else "online"} \\
            --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-final.pt"}'
    """
    # --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-step-20000.pt"}
        

    command = add_environment_variables(command)
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id




def execute_training_multinode(
    *,
    name,
    submit=False,
    test=True,
    array=None,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = {
        # "cpus_per_task": 96,
        # "mem_per_cpu": "1950M",
        "nodes": 2,
        "ntasks_per_node": 1,
        "mem_per_cpu": 2900,
        "cpus_per_task": 288,
        "time": "0:30:00",
        "partition": "debug",
        "gres": "gpu:4",
        "account": "a0186",
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    # srun --time=2:0:0 -n1 -c32 --mem-per-cpu=1950 --gpus-per-task=1 -A a0186 --mpi=pmix --network=disable_rdzv_get --environment=./edf.toml --pty bash -c "cd $prev_home; exec bash --rcfile /capstor/scratch/cscs/tomaszk/.bashrc -i"
    command = (
    "srun --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get "
    "bash -lc "
    "'"
    "cd $SLURM_SUBMIT_DIR; "
    """
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \\
    MASTER_PORT=29500 \\
    RANK=${SLURM_PROCID} \\
    LOCAL_RANK=${SLURM_LOCALID} \\
    WORLD_SIZE=${SLURM_NTASKS}  \\
    """
    """
    echo "MASTER_ADDR=${MASTER_ADDR}";
    echo "MASTER_PORT=${MASTER_PORT}";
    echo "RANK=${RANK}";
    echo "LOCAL_RANK=${LOCAL_RANK}";
    echo "WORLD_SIZE=${WORLD_SIZE}";
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}";
    uv run torchrun --nnodes=2 --nproc-per-node=4 --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \\
    --rdzv-backend=c10d --rdzv-id=${SLURM_JOB_ID} -m euclid_multiprobe_deeplss_training.cli """
    f"""--config={LAUNCH_DIR}/config_deeplss.yaml \\
    --verbosity=info \\
    train \\
    --tag={name}  \\
    --wandb-mode={'online' if submit else 'offline'} \\
    --resume-from-checkpoint={LAUNCH_DIR / 'results' / name / 'checkpoint-final.pt'} 
    """
    "'"
    )
    # --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-step-20000.pt"}
    #     torchrun "
    # "--nnodes=2 "
    # "--nproc-per-node=4 "
    # "--rdzv-backend=c10d "
    # "--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} "
    # "--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} " 
    # "--rdzv-id=${SLURM_JOB_ID} "
    # "-m euclid_multiprobe_deeplss_training.cli"

    command = add_environment_variables(command)
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id


########################################################################################
########################################################################################
##
##
## Pipeline
##
##
########################################################################################
########################################################################################

# Generate the pixel file
# srun pixi run uv run jupyter nbconvert --to notebook --execute repos/euclid-multiprobe-simulation-forward-model/notebooks/pixel_file.ipynb --inplace

# Generate the noise file
# srun pixi run uv run jupyter nbconvert --to notebook --execute repos/euclid-multiprobe-simulation-forward-model/notebooks/noise_file.ipynb --inplace

# Make shell permutation tables
# pixi run uv run python -m cosmogridv11.apps.run_paramtables shell_permutations --config=config_euclidRR2v2multi.yaml   --dir_out=euclidRR2v2multi/ --verbosity=debug
submit_paramtables(name="//permtables",
                   config="config_cosmogridv11_EuclidDR1F.yaml", 
                   dir_out="EuclidDR1F_cosmogridv11")

# Make projected probe maps
# srun pixi run uv run mprof run --interval 0.01 python -m cosmogridv11.apps.run_probemaps main --config=config_cosmogridv11_EuclidDR1F.yaml --dir_out=EuclidDR1F_cosmogridv11 --num_maps_per_index=10 --indices="17" --verbosity=info
submit_probemaps(name="//euclid_proj_test", 
                 config="config_cosmogridv11_EuclidDR1F.yaml", 
                 dir_out="EuclidDR1F_cosmogridv11", 
                 array=[0]+list(range(17, 32)))

submit_probemaps(name="//proj_part4", 
                 config="config_cosmogridv11_EuclidDR1F.yaml", 
                 dir_out="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/", 
                 array=range(500,1000))


# Make tfrecords for probe deep learning training
# srun pixi run uv run mprof run --interval 0.01 python -m msfm.apps.run_onthefly_postprocessing wds --n_files=15 --config=config_msfm_EuclidDR1F_onthefly_test.yaml --dir_in=../000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/ --dir_out=webdataset_EuclidDR1F_onthefly_test/ --cosmogrid_version="1.1" --indices='0' --max_sleep=1 --verbosity=debug
submit_postprocessing(name="//webdataset_test2", 
                 config="config_msfm_EuclidDR1F_onthefly.yaml", 
                 dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/", 
                 dir_out="webdataset_EuclidDR1F_test2", 
                 profile=True,
                 submit=False,
                 array=[0])
# Run training

submit_postprocessing(name="//webdataset_part3", 
                 config="config_msfm_EuclidDR1F_onthefly.yaml", 
                 dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/", 
                 dir_out="webdataset_EuclidDR1F", 
                 array=range(20,50),
                 submit=True)

execute_training_multinode(name="training_ddp_dim256_clariden", 
                        #   array=range(2),
                          submit=True)
