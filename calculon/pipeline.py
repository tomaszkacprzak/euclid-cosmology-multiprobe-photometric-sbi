#!/usr/bin/env python3
import os, glob
from pathlib import Path
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

def run_postprocessing(
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


def run_training(
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
        "cpus_per_task": 48,
        "mem_per_cpu": "3900M",
        "time": "24:00:00",
        "partition": "h200",
        "gpus": 1,
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    command = f"""
    srun --verbose --gpu-bind=none  \\
    pixi run uv run euclid-deeplss-training \\
            --config="{LAUNCH_DIR}/{config}" \\
            --verbosity=info \\
            train \\
            --tag={name} \\
            --wandb-mode={"online" if submit else "online"} \\
            --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-latest.pt"}
    """
        # 


    command = add_environment_variables(command)
    
    if submit:
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
    **kwargs,
):
    if name.startswith("//"):  return None

    slurm_args = {
        "cpus_per_task": 96,
        "mem_per_cpu": "3900M",
        "time": "24:00:00",
        "partition": "h200",
        "nodes": 1,
        "ntasks": 1,
        "gres": "gpu:2",
    } | kwargs

    array = array_string(array, max_running=1) if array is not None else None
    if array is not None:
        slurm_args["array"] = array

    command = f"""
    srun --verbose pixi run uv run 
            torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
            -m euclid_multiprobe_deeplss_training.cli \\
            --config="{LAUNCH_DIR}/config_deeplss.yaml" \\
            --verbosity=info \\
            train \\
            --tag={name} \\
            --wandb-mode={"online" if submit else "online"} \\
            --resume-from-checkpoint={LAUNCH_DIR / "results" / name / "checkpoint-final.pt"}
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



def run_predicton_parallel(
    *,
    name,
    network,
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
                --checkpoint={LAUNCH_DIR / "results" / network / "checkpoint-latest.pt"} \\
                --output-file={LAUNCH_DIR / "results" / network / "preds.h5"} \\
                --num-examples={'20000' if test else '200000'} \\
                --batch-size=64 \\
                --device=cuda \\
        """
    
    if submit:

        slurm_args = {
            "nodes": 1,
            "exclusive": False,
            "mem_per_cpu": 3900,
            "cpus_per_task": 48,
            "time": "12:00:00",
            "partition": "h200",
            "gres": "gpu:1",
            "ntasks_per_node": 1,
        } | kwargs
            
        array = array_string(array, max_running=1) if array is not None else None
        if array is not None:
            slurm_args["array"] = array

        command = f"""
        srun --verbose \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                pixi run uv run  \\
                python {command}
                '
                """

        command = add_environment_variables(command)
    
        job_id = submit_job(slurm_args, name, command)
    
      
    else:


        command = f"""
                    pixi run uv run  \\
                    python {command}
                    """


        command = add_environment_variables(command)
        
        print(command)
        os.system(command)
        job_id = None

    return job_id



def run_likelihood(
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
        "mem_per_cpu": 3900,
        "cpus_per_task": 48,
        "time": "12:00:00",
        "partition": "h200",
        "gres": "gpu:1",
        "ntasks_per_node": 1,
    } | kwargs

    command = f"""-m euclid_multiprobe_deeplss_training.cli \\
                --config="{config}" \\
                --verbosity=info \\
                likelihood \\
                --input-file={LAUNCH_DIR / "results" / network / "preds.h5"} \\
                --output-file={LAUNCH_DIR / "results" / name / "likelihood.pt"} \\
        """
    
    if submit:
        
        array = array_string(array, max_running=1) if array is not None else None
        if array is not None:
            slurm_args["array"] = array

        command = f"""
        srun --verbose  \\
                bash -lc 'cd "$SLURM_SUBMIT_DIR" && \\
                pixi run uv run  \\
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

# Generate the pixel file
# srun pixi run uv run jupyter nbconvert --to notebook --execute repos/euclid-multiprobe-simulation-forward-model/notebooks/pixel_file.ipynb --inplace

# Generate the noise file
# srun pixi run uv run jupyter nbconvert --to notebook --execute repos/euclid-multiprobe-simulation-forward-model/notebooks/noise_file.ipynb --inplace

# Make shell permutation tables
# pixi run uv run python -m cosmogridv11.apps.run_paramtables shell_permutations --config=config_euclidRR2v2multi.yaml   --dir_out=euclidRR2v2multi/ --verbosity=debug
run_paramtables(name="//permtables",
                   config="config_cosmogridv11_EuclidDR1F.yaml", 
                   dir_out="EuclidDR1F_cosmogridv11")

# Compute probemaps

run_probemaps(name="//euclid_proj_test", 
                 config="config_cosmogridv11_EuclidDR1F.yaml", 
                 dir_out="EuclidDR1F_cosmogridv11", 
                 array=[0]+list(range(17, 32)))

run_probemaps(name="//proj_part4", 
                 config="config_cosmogridv11_EuclidDR1F.yaml", 
                 dir_out="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/", 
                 array=range(500,1000))

# Compute webdataset

run_webdataset(name="//webdataset_EuclidDR1F_nside512", 
                   config="conf_nside512_EuclidDR1F.yaml,conf_nside512_mse_corrswin.yaml",
                   dir_in="/scratch/tomaszk/260205_euclid_multiprobe_sbi/000_deeplss_forecast/EuclidDR1F_cosmogridv11/",
                   dir_out="webdataset_EuclidDR1F_nside512",
                   submit=False)

# Training

run_training(name="//corrswin_nside512_mse", 
             config="conf_nside512_mse_corrswin.yaml,conf_nside512_EuclidDR1F.yaml",
             array=range(10,20),
             submit=True)

run_training(name="//mapnvit_nside512_mse", 
             config="conf_nside512_mse_mapnvit.yaml,conf_nside512_EuclidDR1F.yaml",
             array=range(10,20),
             submit=True)           


run_predicton_parallel(name="//predict_corrswin_nside512_mse", 
                       network="corrswin_nside512_mse",
                       config="conf_nside512_mse_corrswin.yaml,conf_nside512_EuclidDR1F.yaml",
                       submit=True)

run_predicton_parallel(name="//predict_mapnvit_nside512_mse", 
                       network="mapnvit_nside512_mse",
                       config="conf_nside512_mse_mapnvit.yaml,conf_nside512_EuclidDR1F.yaml",
                       submit=True)


run_likelihood(name="likelihood_corrswin_nside512_mse", 
                       network="corrswin_nside512_mse",
                       config="conf_nside512_mse_corrswin.yaml,conf_nside512_EuclidDR1F.yaml",
                       submit=True)

run_likelihood(name="//likelihood_mapnvit_nside512_mse", 
                       network="mapnvit_nside512_mse",
                       config="conf_nside512_mse_mapnvit.yaml,conf_nside512_EuclidDR1F.yaml",
                       submit=True)