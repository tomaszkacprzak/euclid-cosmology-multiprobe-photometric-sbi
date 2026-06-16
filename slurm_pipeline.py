#!/usr/bin/env python3
import os, glob
from pathlib import Path
from simple_slurm import Slurm


# Equivalent to workflow.launchDir
LAUNCH_DIR = Path(os.getcwd())
LOG_DIR = LAUNCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

def submit(slurm_args, name, command):

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
        
    job_id = submit(slurm_args, name, command)
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
    
    job_id = submit(slurm_args, name, command)
    return job_id

def submit_postprocessing(
    *,
    name,
    config,
    dir_in,
    dir_out,
    n_files,
    profile=False,
    **kwargs,
):

    if name.startswith("//"):  return None

    slurm_args = {
        "cpus_per_task": 6,
        "mem_per_cpu": "1950M",
        "time": "24:00:00",
        "partition": "cpu-daily",
        "clusters": "calc-cpu",
    } | kwargs
    
    command = f"""
    pixi run uv run {f"mprof run --interval 0.02 " if profile else ""} python -m msfm.apps.run_onthefly_postprocessing wds \\
        --n_files={n_files} \\
        --config="{LAUNCH_DIR / config}" \\
        --dir_in={LAUNCH_DIR / dir_in} \\
        --dir_out={LAUNCH_DIR / dir_out} \\
        --cosmogrid_version="1.1" \\
        --indices="$SLURM_ARRAY_TASK_ID" \\
        --verbosity={"debug" if profile else "info"} 
        --max_sleep={1 if profile else 120}
    """
    
    job_id = submit(slurm_args, name, command)
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

submit_probemaps(name="//proj_part2", 
                 config="config_cosmogridv11_EuclidDR1F.yaml", 
                 dir_out="EuclidDR1F_cosmogridv11", 
                 array=range(32,67))


# Make tfrecords for probe deep learning training
# srun pixi run uv run mprof run --interval 0.01 python -m msfm.apps.run_onthefly_postprocessing postprocess --n_files=15 --config=config_msfm_EuclidDR1F_onthefly_test.yaml --dir_in=../000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/ --dir_out=webdataset_EuclidDR1F_onthefly_test/ --cosmogrid_version="1.1" --indices='0' --max_sleep=1 --verbosity=debug
submit_postprocessing(name="//webdataset_test", 
                 config="config_msfm_EuclidDR1F_onthefly.yaml", 
                 dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/", 
                 dir_out="webdataset_EuclidDR1F_test", 
                 n_files=1,
                 profile=True,
                 array=[0])

submit_postprocessing(name="webdataset_part1", 
                 config="config_msfm_EuclidDR1F_onthefly.yaml", 
                 dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/CosmoGrid/bary/", 
                 dir_out="webdataset_EuclidDR1F", 
                 n_files=51,
                 array=[0] + list(range(17, 50)))