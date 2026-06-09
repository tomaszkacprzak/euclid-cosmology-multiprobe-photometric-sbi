#!/usr/bin/env python3
import os, glob
from pathlib import Path
from simple_slurm import Slurm


# Equivalent to workflow.launchDir
LAUNCH_DIR = Path(os.getcwd())
LOG_DIR = LAUNCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = LAUNCH_DIR / "config_euclidRR2v2multi.yaml"
DIR_OUT = LAUNCH_DIR / "euclidRR2v2multi"

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

def submit(slurm, name, command):

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
    **kwargs,
):

    slurm_args = {
        "cpus_per_task": 1,
        "mem": "15600M",
        "time": "1:00:00",
        "partition": "performance",
        "clusters": "cluster",
        "array": [0],
    }
    slurm_args.update(kwargs)
    slurm_args.update(get_logs(name))
    slurm = Slurm(**slurm_args)

    
    command = f"""
    pixi run uv run python -m cosmogridv11.apps.run_paramtables shell_permutations \\
        --config="{CONFIG_FILE}" \\
        --dir_out="{DIR_OUT}" \\
        --verbosity=info
    """
        
    job_id = submit(slurm, name, command)
    return job_id



def submit_probemaps(
    *,
    name,
    **kwargs,
):
    slurm_args = {
        "cpus_per_task": 8,
        "mem": "15600M",
        "time": "12:00:00",
        "partition": "performance",
        "clusters": "cluster",
    }
    slurm_args.update(kwargs)
    slurm_args.update(get_logs(name))
    slurm = Slurm(**slurm_args)

    
    command = f"""
    pixi run uv run python -m cosmogridv11.apps.run_probemaps main \\
        --config="{CONFIG_FILE}" \\
        --dir_out="{DIR_OUT}" \\
        --num_maps_per_index=10 \\
        --indices="$SLURM_ARRAY_TASK_ID" \\
        --verbosity=info
    """
    
    
    job_id = submit(slurm, name, command)
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


# pixi run uv run python -m cosmogridv11.apps.run_paramtables shell_permutations --config=config_euclidRR2v2multi.yaml   --dir_out=euclidRR2v2multi/ --verbosity=debug
# permtables_job_id = submit_paramtables(name="permtables")

# pixi run uv run python -m cosmogridv11.apps.run_probemaps main --config=config_euclidRR2v2multi.yaml --dir_out=euclidRR2v2multi --num_maps_per_index=10 --indices="17" --verbosity=info
job_id = submit_probemaps(name="euclid_proj_test", array=[0]+list(range(17, 32)))

