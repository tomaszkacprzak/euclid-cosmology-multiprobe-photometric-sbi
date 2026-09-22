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
    export WANDB_ENTITY="euclid-multiprobe-sbi"; \\
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

def run_paramtables(
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



def run_probemaps(
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

def run_webdataset(
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
    pixi run uv run {f"mprof run --interval 0.02 " if profile else ""}  \\
        euclid-deeplss-training  \\
        --config="{LAUNCH_DIR / config}" \\
        --verbosity="debug" \\
        webdataset \\
        --input-dir={LAUNCH_DIR / dir_in} \\
        --output-dir={LAUNCH_DIR / dir_out} \\
        --indices={"$SLURM_ARRAY_TASK_ID" if submit else "0"} \\
    """
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None
    return job_id



def run_training_interactive(
    *,
    name,
    config="config_test.yaml",
    **kwargs,
):
    if name.startswith("//"):  return None
    
    
    command = f"""
            uv run  \\
            euclid-deeplss-training \\
            --config="{LAUNCH_DIR}/{config}" \\
            --verbosity=info \\
            train \\
            --tag={name} \\
            --wandb-mode=online \\
            --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-latest.pt"}
            """
    # --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-step-20000.pt"}

    command = add_environment_variables(command)    

    print(command)
    os.system(command)

    return None







def run_training(
    *,
    name,
    config,
    submit=False,
    test=True,
    array=None,
    **kwargs,
):
    if name.startswith("//"):  return None

    command =  f"""uv run  euclid-deeplss-training \\
                --config="{config}" \\
                --verbosity=info \\
                train \\
                --tag={name} \\
                --wandb-mode={"online" if submit else "online"} \\
                --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-latest.pt"}"""
            
    if submit:
        
        slurm_args = {
            "nodes": 1,
            "exclusive": False,
            "mem_per_cpu": 2900,
            "cpus_per_task": 128,
            "time": "12:00:00",
            "partition": "normal",
            "gres": "gpu:1",
            "account": "a0158",
            "ntasks_per_node": 1,
        } | kwargs

        array = array_string(array, max_running=1) if array is not None else None
        if array is not None:
            slurm_args["array"] = array

        # srun --time=2:0:0 -n1 -c32 --mem-per-cpu=1950 --gpus-per-task=1 -A a0158 --mpi=pmix --network=disable_rdzv_get --environment=./edf.toml --pty bash -c "cd $prev_home; exec bash --rcfile /capstor/scratch/cscs/tomaszk/.bashrc -i"
        command = f"""
        echo WANDB_ENTITY $WANDB_ENTITY; \\
        set -euo pipefail; \\
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && {command}' \\

        """

        command = add_environment_variables(command)
        job_id = submit_job(slurm_args, name, command)
    
    else:

        print(command)
        os.system(command)
        job_id = None

    return job_id





def run_training_parallel(
    *,
    name,
    submit=False,
    test=True,
    array=None,
    config="config_deeplss.yaml",
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = {
        # "cpus_per_task": 96,
        # "mem_per_cpu": "1950M",
        "nodes": 1,
        "exclusive": False,
        "mem_per_cpu": 2900,
        "cpus_per_task": 128,
        "time": "12:00:00",
        "partition": "normal",
        "gres": "gpu:2",
        "account": "a0158",
        "ntasks_per_node": 1,
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    # srun --time=2:0:0 -n1 -c32 --mem-per-cpu=1950 --gpus-per-task=1 -A a0158 --mpi=pmix --network=disable_rdzv_get --environment=./edf.toml --pty bash -c "cd $prev_home; exec bash --rcfile /capstor/scratch/cscs/tomaszk/.bashrc -i"
    command = f"""
    echo WANDB_ENTITY $WANDB_ENTITY; \\
    set -euo pipefail; \\
    srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
            bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
            uv run  \\
            torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
            -m euclid_multiprobe_deeplss_training.cli \\
            --config="{config}" \\
            --verbosity=info \\
            train \\
            --tag={name} \\
            --wandb-mode={"online" if submit else "online"} \\
            --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-latest.pt"}'
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




def run_training_multinode(
    *,
    name,
    submit=False,
    test=True,
    array=None,
    config="config_deeplss.yaml",
    num_nodes=1,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = {
        # "cpus_per_task": 96,
        # "mem_per_cpu": "1950M",
        "nodes": num_nodes,
        "ntasks_per_node": 1,
        "mem_per_cpu": 2900,
        "cpus_per_task": 288,
        "time": "8:0:0",
        "partition": "normal",
        "gres": "gpu:4",
        "account": "a0158",
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    # do not remove the empty line before uv run
    command = (
    f"""
    srun --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get bash -lc ' cd $SLURM_SUBMIT_DIR; 
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
    export MASTER_PORT=29500;
    export RANK=${{SLURM_PROCID}};
    export LOCAL_RANK=${{SLURM_LOCALID}};
    export WORLD_SIZE=${{SLURM_NTASKS}};
    uv run torchrun --nnodes={num_nodes} --nproc-per-node=4 \\
    --rdzv-endpoint=${{MASTER_ADDR}}:${{MASTER_PORT}} --rdzv-backend=c10d --rdzv-id=${{SLURM_JOB_ID}} \\
    -m euclid_multiprobe_deeplss_training.cli \\
    --config={LAUNCH_DIR/config} \\
    --verbosity=info \\
    train \\
    --tag={name}  \\
    --wandb-mode={'online' if submit else 'offline'} \\
    --resume-from-checkpoint={LAUNCH_DIR / 'results' / name / 'checkpoint-latest.pt'}' \\
    """
    )

    command = add_environment_variables(command)
    
    if submit:
        job_id = submit_job(slurm_args, name, command)
    else:
        print(command)
        os.system(command)
        job_id = None

    return job_id



def run_predicton_parallel(
    *,
    name,
    submit=False,
    test=False,
    array=None,
    config="config_deeplss.yaml",
    **kwargs,
):
    if name.startswith("//"):  return None
    


    command = f"""-m euclid_multiprobe_deeplss_training.cli \\
                --config="{config}" \\
                --verbosity=info \\
                predict \\
                --checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-latest.pt"} \\
                --output-file={LAUNCH_DIR / "results" / name / "preds.h5"} \\
                --num-examples={'20000' if test else '200000'} \\
                --batch-size=64 \\
                --device=cuda \\
        """
    
    if submit:

        slurm_args = {
            "nodes": 1,
            "exclusive": False,
            "mem_per_cpu": 2900,
            "cpus_per_task": 128,
            "time": "12:00:00",
            "partition": "normal",
            "gres": "gpu:2",
            "account": "a0158",
            "ntasks_per_node": 1,
        } | kwargs
            
        array = array_string(array, max_running=1) if array is not None else None
        if array is not None:
            slurm_args["array"] = array

        command = f"""
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run  \\
                torchrun --standalone --nnodes=1 --nproc-per-node=2 {command}
                '
                """

        command = add_environment_variables(command)
    
        job_id = submit_job(slurm_args, name, command)
    
      
    else:


        command = f"""
                    uv run  \\
                    torchrun --standalone --nnodes=1 --nproc-per-node=1 {command}
                    """


        command = add_environment_variables(command)
        
        print(command)
        os.system(command)
        job_id = None

    return job_id

def run_likelihood_parallel(
    *,
    name,
    config,
    network,
    submit=False,
    test=False,
    array=None,
    **kwargs,
):
    if name.startswith("//"):  return None
    
    slurm_args = {
        "nodes": 1,
        "exclusive": False,
        "mem_per_cpu": 2900,
        "cpus_per_task": 128,
        "time": "12:00:00",
        "partition": "normal",
        "gres": "gpu:2",
        "account": "a0158",
        "ntasks_per_node": 1,
    } | kwargs

    command = f"""-m euclid_multiprobe_deeplss_training.cli \\
                --config="{config}" \\
                --verbosity=info \\
                likelihood \\
                --input-file={LAUNCH_DIR / "results" / network / "preds.h5"} \\
                --output-file={LAUNCH_DIR / "results" / name / "likelihood.pt"} \\
                --num-observations=10 \\
        """
    
    if submit:
        
        array = array_string(array, max_running=1) if array is not None else None
        if array is not None:
            slurm_args["array"] = array

        command = f"""
        srun -K1 --verbose --environment=./edf.toml --mpi=pmix --network=disable_rdzv_get \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                uv run  \\
                python {command} \\
                '
                """

        command = add_environment_variables(command)
    
        job_id = submit_job(slurm_args, name, command)
    
      
    else:


        command = f"""
                    nvidia-smi; \\
                    srun --partition=h200 --time=0:30:0 --gpus=1 --cpus-per-task=32 --mem-per-cpu=1950 --verbose --gpu-bind=none \\
                    pixi run uv run  \\
                    {command}
                    """

        command = add_environment_variables(command)
        
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
                 

# webdataset

run_webdataset(name="//webdataset_EuclidDR1F_nside512", 
                   config="conf_nside512_EuclidDR1F.yaml,conf_nside512_mse_corrswin.yaml",
                   dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/",
                   dir_out="webdataset_EuclidDR1F_nside512",
                   submit=False)

# training 

run_training_parallel(name="//corrswin_nside512_mse", 
                          config="conf_nside512_mse_corrswin.yaml",
                          array=range(0,10),
                          submit=True)

run_training_parallel(name="//mapnvit_nside512_mse", 
                          config="conf_nside512_mse_mapnvit.yaml",
                          array=range(0,10),
                          submit=True)

# prediction

run_predicton_parallel(name="//corrswin_nside512_mse", 
                           config="conf_nside512_mse_corrswin.yaml",
                           test=True,
                           submit=False)

run_predicton_parallel(name="//mapnvit_nside512_mse", 
                           config="conf_nside512_mse_mapnvit.yaml",
                           submit=False)

run_likelihood_parallel(name="likelihood_corrswin_nside512_mse", 
                        network="corrswin_nside512_mse",
                        config="conf_nside512_mse_corrswin.yaml,config_nside512_EuclidDR1F.yaml",
                        submit=False)